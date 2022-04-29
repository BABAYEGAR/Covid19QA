import abc

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

N_TOP = 20


class Retrieval(abc.ABC):
    def __init__(self, docs, keys=None):
        self.docs = docs.copy()
        if keys is not None:
            self.docs.index = keys
        self.model = None
        self.vectorizer = None

    def top_n_documents(self, vector):
        similarity = cosine_similarity(self.vectorizer, vector)
        rankings = np.argsort(np.squeeze(similarity))[::-1]
        ranked_indices = self.docs.index[rankings]
        return self.docs[ranked_indices][:N_TOP]

    @abc.abstractmethod
    def retrieve(self, query):
        pass


class TFIDF(Retrieval):
    def __init__(self, docs, keys=None):
        super(TFIDF, self).__init__(docs, keys)
        self.model = TfidfVectorizer()
        self.vectorizer = self.model.fit_transform(docs)

    def retrieve(self, query):
        return self.top_n_documents(self.model.transform([query]))
