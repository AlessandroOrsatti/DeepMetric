"""Here we can see how to import functions and submodules of our projects."""

# more complex: import submodules if a package is installed
from importlib.util import find_spec

# this line imports all the functions of "utils"

if find_spec("torch"):
    from . import torch_utils

