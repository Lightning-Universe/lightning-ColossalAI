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
from typing import Any, Callable, Literal, Optional, Union

from lightning_utilities import module_available
from torch import Tensor
from torch.optim import Optimizer

if module_available("lightning"):
    from lightning.fabric.utilities.types import Steppable
    from lightning.pytorch import LightningModule
    from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
elif module_available("pytorch_lightning"):
    from lightning_fabric.utilities.types import Steppable
    from pytorch_lightning import LightningModule
    from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")


class ColossalAIPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for ColossalAI integration.

    Args:
        precision: Half precision (16-mixed).

    Raises:
        ValueError:
            If precison is not 16-mixed.
    """

    def __init__(self, precision: Literal["16-mixed"] = 16) -> None:
        if precision != "16-mixed":
            raise ValueError(
                f"`Trainer(strategy='colossalai', precision={precision!r})` is not supported."
                " Consider setting `precision='16-mixed'`."
            )
        self.precision = precision

    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: LightningModule,
        optimizer: Optional[Steppable],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert optimizer is not None
        optimizer.backward(tensor)

    def clip_grad_by_norm(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        optimizer.clip_grad_norm(None, clip_val)

    def clip_grad_by_value(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        raise NotImplementedError("`clip_grad_by_value` is not supported by `ColossalAI`")

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: LightningModule,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        closure_result = closure()
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        if isinstance(model, LightningModule) and model.automatic_optimization and skipped_backward:
            raise ValueError(
                "Skipping backward by returning `None` from your `training_step` is not supported by `ColossalAI`."
            )
        optimizer.step()
