from . import data
from . import losses
from . import vb_layers

from . import engine
from . import applications

# Load FCN models
from .applications.simple_fcn import default as simple_fcn
from .applications.simple_fcn import vbranch_default as vbranch_simple_fcn

# Load CNN models
from .applications.simple_cnn import default as simple_cnn
from .applications.simple_cnn import vbranch_default as vbranch_simple_cnn
