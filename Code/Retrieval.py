import abc

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Retrieval(abc.ABC):
    def __init__(self, docs, keys=None):
        self.docs = docs.copy()
        if keys is not None:
            self.docs.index = keys
        self.model = None
        self.vectorizer = None

    def top_n_documents(self, vector, top_count=10):
        similarity = cosine_similarity(self.vectorizer, vector)
        rankings = np.argsort(np.squeeze(similarity))[::-1]
        ranked_indices = self.docs.index[rankings]
        return self.docs[ranked_indices][:top_count]

    @abc.abstractmethod
    def retrieve(self, query, top_count=10):
        pass
