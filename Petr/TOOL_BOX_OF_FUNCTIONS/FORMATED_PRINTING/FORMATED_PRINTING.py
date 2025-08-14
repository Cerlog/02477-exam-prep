import numpy as np
import jax.numpy as jnp

class _FmtVal:
    def __init__(self, x, decimals=2):
        self.x = x
        self.decimals = decimals

    def _to_string(self):
        fmt_str = f"{{:.{self.decimals}f}}"
        x = self.x
        if isinstance(x, (float, np.floating, jnp.floating)):
            return fmt_str.format(float(x))
        elif np.isscalar(x):
            return str(x)
        elif isinstance(x, (np.ndarray, jnp.ndarray, list, tuple)):
            arr = np.array(x)
            return np.array2string(arr, formatter={'float_kind': lambda v: fmt_str.format(v)})
        else:
            return str(x)

    # used by f-strings and format()
    def __format__(self, spec):
        # ignore 'spec' so {ffmt(x):.3f} won't double-format; decimals controls it
        return self._to_string()

    def __str__(self):
        return self._to_string()

def ffmt(x, decimals=2):
    """Use inside f-strings: f\"... {ffmt(x)} ...\"."""
    return _FmtVal(x, decimals=decimals)

def print_fmt(*args, decimals=2, sep=' ', end='\n'):
    """Print like print(), formatting floats/arrays to fixed decimals."""
    fmt_str = f"{{:.{decimals}f}}"

    def format_value(x):
        if isinstance(x, (float, np.floating, jnp.floating)):
            return fmt_str.format(float(x))
        elif np.isscalar(x):
            return str(x)
        elif isinstance(x, (np.ndarray, jnp.ndarray, list, tuple)):
            arr = np.array(x)
            return np.array2string(arr, formatter={'float_kind': lambda v: fmt_str.format(v)})
        else:
            return str(x)

    print(sep.join(format_value(a) for a in args), end=end)