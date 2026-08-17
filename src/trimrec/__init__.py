"""TRIMRec package."""

from trimrec.config import ExperimentConfig, load_config
from trimrec.model.trimrec import TRIMRec, TRIMRecOutput

__all__ = ["ExperimentConfig", "TRIMRec", "TRIMRecOutput", "load_config"]
__version__ = "1.0.0"
