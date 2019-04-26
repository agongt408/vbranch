from .core import Dense, BatchNormalization, Activation, Input, VBOutput
from .convolutional import Conv2D
from .pooling import AveragePooling2D, GlobalAveragePooling2D
from .vb_merge import MergeSharedUnique

# Re-name vb merge layers
from .vb_merge import Add as AddVB
from .vb_merge import Average as AverageVB
from .vb_merge import Concatenate as ConcatenateVB