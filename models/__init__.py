# the bi-encoder model architecture (which consists of a query/a document encoder)
from .inbatch import InBatch
# [todo] try to merge this into the first one.
from .inbatch import InBatchForSplade 
from .lateinteraction import LateInteraction

# the encoder models
from ._contriever import Contriever
from ._splade import SpladeRep
