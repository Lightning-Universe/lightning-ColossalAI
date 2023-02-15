# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytest
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import ColossalAIStrategy
from torch import Tensor

from tests.datamodules import ClassifDataModule
from tests.test_strategy import ModelParallelBoringModel, ModelParallelClassificationModel


def decorate(func, standalone: bool):
    """Mock functions for parsing standalone test, but it does not do anything."""
    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    return wrap


@decorate(standalone=True)
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test needs at least single GPU.")
def test_colossalai_optimizer(tmpdir):
    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=1,
        precision=16,
        strategy="colossalai",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(
        ValueError,
        match="`ColossalAIStrategy` only supports `colossalai.nn.optimizer.CPUAdam` "
        "and `colossalai.nn.optimizer.HybridAdam` as its optimizer.",
    ):
        trainer.fit(model)


@decorate(standalone=True)
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test needs at least single GPU.")
def test_warn_colossalai_ignored(tmpdir):
    class TestModel(ModelParallelBoringModel):
        def backward(self, loss: Tensor, *args, **kwargs) -> None:
            return loss.backward()

    model = TestModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=1,
        precision=16,
        strategy="colossalai",
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with pytest.warns(UserWarning, match="will be ignored since ColossalAI handles the backward"):
        trainer.fit(model)


@decorate(standalone=True)
def _assert_save_model_is_equal(model, tmpdir, trainer):
    checkpoint_path = os.path.join(tmpdir, "model.pt")
    checkpoint_path = trainer.strategy.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    trainer.strategy.barrier()

    # carry out the check only on rank 0
    if trainer.is_global_zero:
        state_dict = torch.load(checkpoint_path)

        # Assert model parameters are identical after loading
        for orig_param, saved_model_param in zip(model.parameters(), state_dict.values()):
            saved_model_param = saved_model_param.to(dtype=orig_param.dtype, device=orig_param.device)
            assert torch.equal(orig_param, saved_model_param)


@decorate(standalone=True)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
def test_multi_gpu_checkpointing(tmpdir):
    dm = ClassifDataModule()
    model = ModelParallelClassificationModel()
    ck = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, save_top_k=-1)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        accelerator="gpu",
        devices=2,
        precision=16,
        strategy="colossalai",
        callbacks=[ck],
        num_sanity_val_steps=0,  # TODO: remove once validation/test before fitting is supported again
    )
    trainer.fit(model, datamodule=dm)

    results = trainer.test(datamodule=dm)
    saved_results = trainer.test(ckpt_path=ck.best_model_path, datamodule=dm)
    assert saved_results == results


@decorate(standalone=True)
@pytest.mark.xfail(raises=AssertionError, match="You should run a completed iteration as your warmup iter")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
def test_test_without_fit(tmpdir):
    model = ModelParallelClassificationModel()
    dm = ClassifDataModule()
    trainer = Trainer(default_root_dir=tmpdir, accelerator="gpu", devices=2, precision=16, strategy="colossalai")

    # Colossal requires warmup, you can't run validation/test without having fit first
    # This is a temporary limitation
    trainer.test(model, datamodule=dm)


@decorate(standalone=True)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
def test_multi_gpu_model_colossalai_fit_test(tmpdir):
    seed_everything(7)

    dm = ClassifDataModule()
    model = ModelParallelClassificationModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=2,
        precision=16,
        strategy=ColossalAIStrategy(initial_scale=32),
        max_epochs=1,
        num_sanity_val_steps=0,  # TODO: remove once validation/test before fitting is supported again
    )
    trainer.fit(model, datamodule=dm)

    if trainer.is_global_zero:
        out_metrics = trainer.callback_metrics
        assert out_metrics["train_acc"].item() > 0.7
        assert out_metrics["val_acc"].item() > 0.7

    result = trainer.test(model, datamodule=dm)
    if trainer.is_global_zero:
        for out in result:
            assert out["test_acc"] > 0.7
