from sklearn.feature_extraction.text import TfidfVectorizer

import Retrieval


class TdifAnalysis(Retrieval.Retrieval):
    """Retrieve documents based on cosine similarity of TF-IDF vectors with query."""
    def __init__(self, docs, keys=None):
        super(TdifAnalysis, self).__init__(docs, keys)
        self.model = TfidfVectorizer()
        self.vectorizer = self.model.fit_transform(docs)

    def retrieve(self, query, top_n=10):
        q_vec = self.model.transform([query])
        return self._top_documents(q_vec, top_n)
