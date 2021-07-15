
from . import recording
from . import analysis
from . import stimuli
from . import tests


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
