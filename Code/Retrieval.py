import abc

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Retrieval(abc.ABC):
    """Base class for retrieval methods."""
    def __init__(self, docs, keys=None):
        self._docs = docs.copy()
        if keys is not None:
            self._docs.index = keys
        self.model = None
        self.vectorizer = None

    def _top_documents(self, q_vec, top_n=10):
        similarity = cosine_similarity(self.vectorizer, q_vec)
        rankings = np.argsort(np.squeeze(similarity))[::-1]
        ranked_indices = self._docs.index[rankings]
        return self._docs[ranked_indices][:top_n]

    @abc.abstractmethod
    def retrieve(self, query, top_n=10):
        pass
