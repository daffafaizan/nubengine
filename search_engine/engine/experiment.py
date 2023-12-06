import os
import math
import re

from collections import defaultdict
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
    score = 0

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
    score = 0

    for i in range(1, len(ranking) + 1):
        score += ranking[i - 1] / math.log(i + 1, 2)

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
    k = min(k, len(ranking))
    score = sum(ranking[:k]) / k

    return score


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
    rsum = 0
    result = 0

    for i in range(1, len(ranking) + 1):
        if ranking[i - 1]:
            rsum += 1
            result += rsum / i
    approxR = sum(ranking)
    if(approxR == 0):
        return 0
    score = result / approxR
    
    return score

# >>>>> memuat qrels


def load_qrels(qrel_file="engine/training/nfcorpus/test.3-2-1.qrel"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    qrel_file = os.path.join(os.getcwd(), qrel_file)

    qrels = defaultdict(lambda: defaultdict(lambda: 0))

    with open(qrel_file, encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = parts[2]
            qrels[qid][did] = 1

    return qrels

# >>>>> EVALUASI !


def eval_retrieval(qrels, query_file="engine/training/nfcorpus/test.all.queries", k=1, use_letor=False):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    
    BSBI_instance.load()

    query_file = os.path.join(os.getcwd(), query_file)

    with open(query_file, encoding="utf-8") as file:
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []

        bm25_parameters = [
            (1.2, 0.9),
        ]

        rbp_scores_bm25 = [[] for _ in range(len(bm25_parameters))]
        dcg_scores_bm25 = [[] for _ in range(len(bm25_parameters))]
        ap_scores_bm25 = [[] for _ in range(len(bm25_parameters))]

        if use_letor:
            letor = Letor()

        for qline in tqdm(file):
            qid, query = qline.split("\t")

            """
            Evaluasi TF-IDF
            """
            ranking_tfidf = []
            tf_idf_result = BSBI_instance.retrieve_tfidf(query, k = k)

            if use_letor:
                tf_idf_result = letor.rerank(query, [t[1] for t in tf_idf_result])

            for (score, doc) in tf_idf_result:
                did = (re.split(r'[\\/\.]', doc)[-2])
                if (did in qrels[qid]):
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)

            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))

            """
            Evaluasi BM25
            """
            for i, (k1, b) in enumerate(bm25_parameters):
                ranking_bm25 = []
                bm_25_result = BSBI_instance.retrieve_bm25(query, k=k, k1=k1, b=b)
                if use_letor:
                    bm_25_result = letor.rerank(query, [t[1] for t in bm_25_result])
                for (score, doc) in bm_25_result:
                    did = (re.split(r'[\\/\.]', doc)[-2])
                    if (did in qrels[qid]):
                        ranking_bm25.append(1)
                    else:
                        ranking_bm25.append(0)

                rbp_scores_bm25[i].append(rbp(ranking_bm25))
                dcg_scores_bm25[i].append(dcg(ranking_bm25))
                ap_scores_bm25[i].append(ap(ranking_bm25))

    print("Hasil evaluasi TF-IDF terhadap 150 queries")
    print(f"RBP score = {sum(rbp_scores_bm25[i]) / len(rbp_scores_bm25[i]):>.4f}")
    print(f"DCG score = {sum(dcg_scores_bm25[i]) / len(dcg_scores_bm25[i]):>.4f}")
    print(f"AP score  = {sum(ap_scores_bm25[i]) / len(ap_scores_bm25[i]):>.4f}")
    print("")

    for i, (k1, b) in enumerate(bm25_parameters):
        print(f"Hasil evaluasi BM25 (k1={k1}, b={b}) terhadap 150 queries")
        print(f"RBP score = {sum(rbp_scores_bm25[i]) / len(rbp_scores_bm25[i]):>.4f}")
        print(f"DCG score = {sum(dcg_scores_bm25[i]) / len(dcg_scores_bm25[i]):>.4f}")
        print(f"AP score  = {sum(ap_scores_bm25[i]) / len(ap_scores_bm25[i]):>.4f}")
        print()


if __name__ == '__main__':
    qrels = load_qrels()

    print("Sebelum Letor")
    eval_retrieval(qrels)
    
    print("Setelah Letor (LambdaMart)")
    eval_retrieval(qrels, use_letor = True)
