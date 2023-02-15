import os

from pl_colossalai.__about__ import *  # noqa: F401, F403
from pl_colossalai.precision import ColossalAIPrecisionPlugin
from pl_colossalai.strategy import ColossalAIStrategy

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)


__all__ = ["ColossalAIStrategy", "ColossalAIPrecisionPlugin"]
