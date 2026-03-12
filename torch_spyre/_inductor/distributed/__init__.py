from .lowerings import register_lowerings
from .fx_pass import lower_collectives


# Import kernels so dispatcher registrations happen
from . import kernels

register_lowerings()