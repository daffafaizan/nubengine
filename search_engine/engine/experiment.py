from collections import defaultdict
import math
import re
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm

from letor import Letor

# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        p = 1 / math.log(i+1, 2)
        score += p * ranking[pos]
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    # TODO
    if k == None:
        k = len(ranking)
    score = 0.
    for i in range(1, k + 1):
        pos = i - 1
        score += ranking[pos]
    return score/k


def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    # TODO
    R = 0
    for i in range(len(ranking)):
        R += ranking[i]
    if R == 0:
        return 0
    score = 0.
    for i in range(len(ranking)):
        prec_score = prec(ranking, i+1)
        score += prec_score * ranking[i]
    return score/R


# >>>>> memuat qrels


def load_qrels(qrel_file="nfcorpus/test.3-2-1.qrel"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """

    qrels = defaultdict(lambda: defaultdict(lambda: 0))
    with open(qrel_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


# >>>>> EVALUASI !


def eval_retrieval(qrels, query_file="nfcorpus/test.all.queries", k=100):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.load()

    with open(query_file, 'r', encoding='utf-8') as file:
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []

        rbp_scores_bm25 = []
        dcg_scores_bm25 = []
        ap_scores_bm25 = []

        rbp_scores_tfidf_ltr = []
        dcg_scores_tfidf_ltr = []
        ap_scores_tfidf_ltr = []

        rbp_scores_bm25_ltr = []
        dcg_scores_bm25_ltr = []
        ap_scores_bm25_ltr = []

        ltr = Letor()
        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            Evaluasi TF-IDF
            """
            result_tfidf = BSBI_instance.retrieve_tfidf(query, k=k)

            ranking_tfidf = []

            for (score, doc) in result_tfidf:
                did = int(re.split(r'[\\/\.]', doc)[-2])
                print(did)
                ranking_tfidf.append(qrels[qid][did])

            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))

            """
            Evaluasi TF-IDF DAN LETOR
            """
            docs_tfidf = []
            ranking_tfidf_ltr = []
            for doc in [x[1] for x in result_tfidf]:
                with open(doc, encoding="utf-8") as file:
                    text = file.read()
                docs_tfidf.append((doc, text))
            scores = ltr.predict(docs_tfidf, query)
            sorted_did_scores = ltr.evaluate(docs_tfidf, scores)

            for doc, score in sorted_did_scores:
                did = int(re.split(r'[\\/\.]', doc)[-2])
                ranking_tfidf_ltr.append(qrels[qid][did])

            rbp_scores_tfidf_ltr.append(rbp(ranking_tfidf_ltr))
            dcg_scores_tfidf_ltr.append(dcg(ranking_tfidf_ltr))
            ap_scores_tfidf_ltr.append(ap(ranking_tfidf_ltr))

            """
            Evaluasi BM25
            """
            result_bm25 = BSBI_instance.retrieve_bm25(query, k=k)

            ranking_bm25 = []
            for (score, doc) in result_bm25:
                did = int(re.split(r'[\\/\.]', doc)[-2])
                ranking_bm25.append(qrels[qid][did])

            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))

            """
            Evaluasi BM-25 DAN LETOR
            """
            docs_bm25 = []
            ranking_bm25_ltr = []
            for doc in [x[1] for x in result_bm25]:
                with open(doc, encoding="utf-8") as file:
                    text = file.read()
                docs_bm25.append((doc, text))
            scores = ltr.predict(docs_bm25, query)
            sorted_did_scores = ltr.evaluate(docs_bm25, scores)

            for doc, score in sorted_did_scores:
                did = int(re.split(r'[\\/\.]', doc)[-2])
                ranking_bm25_ltr.append(qrels[qid][did])

            rbp_scores_bm25_ltr.append(rbp(ranking_bm25_ltr))
            dcg_scores_bm25_ltr.append(dcg(ranking_bm25_ltr))
            ap_scores_bm25_ltr.append(ap(ranking_bm25_ltr))

    print("Hasil evaluasi TF-IDF terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))

    print("Hasil evaluasi TF-IDF DAN LETOR terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf_ltr) / len(rbp_scores_tfidf_ltr))
    print("DCG score =", sum(dcg_scores_tfidf_ltr) / len(dcg_scores_tfidf_ltr))
    print("AP score  =", sum(ap_scores_tfidf_ltr) / len(ap_scores_tfidf_ltr))

    print("Hasil evaluasi BM25 terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
    print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
    print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))

    print("Hasil evaluasi BM25 DAN LETOR terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25_ltr) / len(rbp_scores_bm25_ltr))
    print("DCG score =", sum(dcg_scores_bm25_ltr) / len(dcg_scores_bm25_ltr))
    print("AP score  =", sum(ap_scores_bm25_ltr) / len(ap_scores_bm25_ltr))


if __name__ == '__main__':
    qrels = load_qrels()

    eval_retrieval(qrels)
