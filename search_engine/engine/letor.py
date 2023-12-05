import os
import re
import random

import lightgbm as lgb
import numpy as np

from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class Letor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Letor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.documents = {}
        self.queries = {}
        self.dataset = []

        self.stemmer_eng = PorterStemmer()
        self.stop_words_eng = set(stopwords.words('english'))

        self.stemmer_indo = MPStemmer()
        self.stop_word_remover_indo = StopWordRemoverFactory().create_stop_word_remover()

        self.load_train_data()
        self.create_dataset()
        self.build_model()
        self.train()

    def preprocess_eng(self, line: str) -> list[str]:
        stemmed_line = " ".join([self.stemmer_eng.stem(word) for word in re.findall(r'\w+', line) if word.lower() not in self.stop_words_eng])

        return re.findall(r'\w+', stemmed_line)
    
    def preprocess_indo(self, line: str) -> list[str]:
        stemmed_line: str = self.stemmer_indo.stem_kalimat(line)
        preprocessed_line: str = self.stop_word_remover_indo.remove(stemmed_line)

        return re.findall(r'\w+', preprocessed_line)

    def load_train_data(self):
        # English
        docs_file_eng = os.path.join(os.getcwd(), "engine/training/nfcorpus/train.docs")
        query_file_eng = os.path.join(os.getcwd(), "engine/training/nfcorpus/train.vid-desc.queries")
        # Indo
        docs_file_indo = os.path.join(os.getcwd(), "engine/experiments/letor/train_docs.txt")
        query_file_indo = os.path.join(os.getcwd(), "engine/experiments/letor/train_queries.txt")

        with open(docs_file_eng) as file:
            for line in file:
                doc_id, content = line.split("\t") #nfcorpus
                self.documents[doc_id] = self.preprocess_eng(content)
        with open(docs_file_indo) as file:
            for line in file:
                doc_id, content = line.strip().split(maxsplit=1)
                self.documents[doc_id] = self.preprocess_indo(content)

        with open(query_file_eng, encoding="utf-8") as file:
            for line in file:
                q_id, content = line.split("\t") #nfcorpus
                self.queries[q_id] = self.preprocess_eng(content)
        with open(query_file_indo, encoding="utf-8") as file:
            for line in file:
                q_id, content = line.strip().split(maxsplit=1)
                self.queries[q_id] = self.preprocess_indo(content)

    def create_dataset(self, NUM_NEGATIVES=1):
        # grouping by q_id first
        # English
        qrels_file_eng = os.path.join(os.getcwd(), "engine/training/nfcorpus/train.3-2-1.qrel")
        # Indo
        qrels_file_indo = os.path.join(os.getcwd(), "engine/experiments/letor/train_qrels.txt")

        q_docs_rel = {}

        with open(qrels_file_eng) as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t") #nfcorpus
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in q_docs_rel:
                        q_docs_rel[q_id] = []
                    q_docs_rel[q_id].append((doc_id, int(rel)))

        with open(qrels_file_indo) as file:
            for line in file:
                q_id, doc_id, rel = line.split()
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
            max_depth=-1
        )
        self.ranker.fit(X, Y, group=self.group_qid_count)

    def predict(self, query, docs):
        if not docs:
            return []

        # bentuk ke format numpy array
        X_unseen = []

        for doc in docs:
            with open(doc, "r", encoding="utf-8") as file:
                X_unseen.append(self.features(self.preprocess_eng(query), self.preprocess_eng(file.readline())))

        X_unseen = np.array(X_unseen)
        self.scores = self.ranker.predict(X_unseen)

        return self.scores

    def rerank(self, query, docs):
        scores = self.predict(query, docs)
        did_scores = [x for x in zip(scores, docs)]
        sorted_did_scores = sorted(
            did_scores, key=lambda tup: tup[0], reverse=True)
        
        return sorted_did_scores