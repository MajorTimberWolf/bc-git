import functools

from vizier._src.jax.optimizers.core import LossFunction
from vizier._src.jax.optimizers.core import Optimizer

@functools.lru_cache
def default_optimizer(maxiter: int = 50) -> Optimizer:
  """Default optimizer and random restarts that work okay for most cases."""
  # NOTE: Production algorithms are recommended to stay away from using this.
  return JaxoptScipyLbfgsB(
    LbfgsBOptions(maxiter=maxiter, best_n=None)
) #IGNORE: ignore error lines