import os
import random

import lightgbm as lgb
import numpy as np
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine


class Letor:
    def __init__(self) -> None:
        self.documents = {}
        self.queries = {}
        self.dataset = []

        self.load_data()
        self.create_dataset()
        self.build_model()
        self.train()

    def load_data(self):
        with open("engine/nfcorpus/train.docs") as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()
        with open("engine/nfcorpus/train.vid-desc.queries", encoding="utf-8") as file:
            for line in file:
                q_id, content = line.split("\t")
                self.queries[q_id] = content.split()

    def create_dataset(self, NUM_NEGATIVES=1):
        # grouping by q_id first
        q_docs_rel = {}
        with open("engine/nfcorpus/train.3-2-1.qrel") as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in q_docs_rel:
                        q_docs_rel[q_id] = []
                    q_docs_rel[q_id].append((doc_id, int(rel)))

        # group_qid_count untuk model LGBMRanker
        self.group_qid_count = []
        self.dataset = []
        for q_id in q_docs_rel:
            docs_rels = q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append(
                    (self.queries[q_id], self.documents[doc_id], rel))
            # tambahkan satu negative (random sampling saja dari documents)
            self.dataset.append(
                (self.queries[q_id], random.choice(list(self.documents.values())), 0))

    def build_model(self, NUM_LATENT_TOPICS=200):
        # bentuk dictionary, bag-of-words corpus, dan kemudian Latent Semantic Indexing
        # dari kumpulan 3612 dokumen.
        self.dict = Dictionary()
        bow_corpus = [self.dict.doc2bow(doc, allow_update=True)
                      for doc in self.documents.values()]
        self.model = LsiModel(bow_corpus, num_topics=NUM_LATENT_TOPICS)

    def vector_rep(self, text, NUM_LATENT_TOPICS=200):
        # test melihat representasi vector dari sebuah dokumen & query

        rep = [topic_value for (_, topic_value)
               in self.model[self.dict.doc2bow(text)]]
        return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def formatting(self):
        X = []
        Y = []

        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def train(self):
        X, Y = self.formatting()
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=100,
            importance_type="gain",
            metric="ndcg",
            num_leaves=40,
            learning_rate=0.02,
            max_depth=-1)
        self.ranker.fit(X, Y, group=self.group_qid_count)

    def predict(self, docs, query):
        if not docs:
            return []

        # bentuk ke format numpy array
        X_unseen = []

        for doc_id, doc in docs:
            X_unseen.append(self.features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)
        self.scores = self.ranker.predict(X_unseen)
        return self.scores

    def evaluate(self, docs, scores):
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(
            did_scores, key=lambda tup: tup[1], reverse=True)
        return sorted_did_scores
