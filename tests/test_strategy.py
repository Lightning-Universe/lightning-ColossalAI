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
import torch.nn.functional as F  # noqa: N812
from colossalai.nn.optimizer import HybridAdam
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins.precision import ColossalAIPrecisionPlugin
from lightning.pytorch.strategies import ColossalAIStrategy
from torch import nn
from torchmetrics import Accuracy


def test_invalid_colosalai(monkeypatch):
    import lightning.pytorch.strategies.colossalai as colossal_strategy

    monkeypatch.setattr(colossal_strategy, "_COLOSSALAI_AVAILABLE", False)
    with pytest.raises(
        ModuleNotFoundError,
        match="To use the `ColossalAIStrategy`, please install `colossalai` first. "
        "Download `colossalai` by consulting `https://colossalai.org/download`.",
    ):
        ColossalAIStrategy()


def test_colossalai_strategy_with_trainer_by_instance():
    trainer = Trainer(precision=16, strategy=ColossalAIStrategy())

    assert isinstance(trainer.strategy, ColossalAIStrategy)
    assert isinstance(trainer.strategy.precision_plugin, ColossalAIPrecisionPlugin)


def test_colossalai_strategy_with_trainer_by_string():
    trainer = Trainer(precision=16, strategy="colossalai")

    assert isinstance(trainer.strategy, ColossalAIStrategy)
    assert isinstance(trainer.strategy.precision_plugin, ColossalAIPrecisionPlugin)


class ModelParallelBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def configure_sharded_model(self) -> None:
        self.layer = torch.nn.Linear(32, 2)

    def configure_optimizers(self):
        optimizer = HybridAdam(self.layer.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class ModelParallelBoringModelNoSchedulers(ModelParallelBoringModel):
    def configure_optimizers(self):
        return HybridAdam(self.layer.parameters(), lr=1e-3)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test needs at least single GPU.")
def test_gradient_clip_algorithm_error(tmpdir):
    model = ModelParallelBoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        accelerator="gpu",
        devices=1,
        precision=16,
        strategy="colossalai",
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
    )
    with pytest.raises(NotImplementedError, match="`clip_grad_by_value` is not supported by `ColossalAI`"):
        trainer.fit(model)


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


class ModelParallelClassificationModel(LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()

        self.lr = lr
        self.layers = None

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

    def build_layers(self) -> nn.Module:
        layers = []
        for _ in range(3):
            layers.append(nn.Linear(32, 32))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 3))
        return nn.Sequential(*layers)

    def configure_sharded_model(self) -> None:
        if self.layers is None:
            self.layers = self.build_layers()

    def forward(self, x):
        x = self.layers(x)
        logits = F.softmax(x, dim=1)
        return logits

    def configure_optimizers(self):
        optimizer = HybridAdam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc(logits, y), prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("val_loss", F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log("val_acc", self.valid_acc(logits, y), prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("test_loss", F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.forward(x)
