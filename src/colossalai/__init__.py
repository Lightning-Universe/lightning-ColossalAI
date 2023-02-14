import os

from colossalai.__about__ import *  # noqa: F401, F403
from colossalai.precision import ColossalAIPrecisionPlugin  # noqa: F401
from colossalai.strategy import ColossalAIStrategy  # noqa: F401

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
