import os

from lightning_colossalai.__about__ import *  # noqa: F401, F403
from lightning_colossalai.precision import ColossalAIPrecisionPlugin  # noqa: F401
from lightning_colossalai.strategy import ColossalAIStrategy  # noqa: F401

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
