"A toolbox for analyzing and processing mobility data."

__version__ = "1.2.3"

from .core.flowdataframe import FlowDataFrame  # noqa
from .core.trajectorydataframe import TrajDataFrame  # noqa
from .io.file import read, write  # noqa
