import importlib
import sys

def reload_module(name):
    """
    Reload a module by its import path string.
    Example:
        reload_module("TOOL_BOX_OF_FUNCTIONS.SAMPLING")
    """
    if name in sys.modules:
        importlib.reload(sys.modules[name])
    else:
        __import__(name)

# example 
#reload_module("TOOL_BOX_OF_FUNCTIONS.SAMPLING")
#from TOOL_BOX_OF_FUNCTIONS.SAMPLING import SAMPLING as SAMP