from . import losses
from . import vb_layers
from . import datasets
from . import engine
from . import applications

# Load FCN models
from .applications.fcn import base as simple_fcn

# Load CNN models
from .applications.cnn import base as simple_cnn

from .applications.resnet import base as resnet
