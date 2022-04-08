from sklearn.feature_extraction.text import TfidfVectorizer

import Retrieval


class TdifAnalysis(Retrieval.Retrieval):
    def __init__(self, docs, keys=None):
        super(TdifAnalysis, self).__init__(docs, keys)
        self.model = TfidfVectorizer()
        self.vectorizer = self.model.fit_transform(docs)

    def retrieve(self, query, top_count=10):
        return self.top_n_documents(self.model.transform([query]), top_count)
