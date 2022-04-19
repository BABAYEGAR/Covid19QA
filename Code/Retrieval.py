import abc

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


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


class TdifAnalysis(Retrieval):
    def __init__(self, docs, keys=None):
        super(TdifAnalysis, self).__init__(docs, keys)
        self.model = TfidfVectorizer()
        self.vectorizer = self.model.fit_transform(docs)

    def retrieve(self, query, top_count=10):
        return self.top_n_documents(self.model.transform([query]), top_count)
