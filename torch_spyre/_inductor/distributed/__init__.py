from . import spyre_library
from . import kernels

from .lowerings import register_lowerings
from .fx_pass import lower_collectives

register_lowerings()