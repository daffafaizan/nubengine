import re
from django.shortcuts import render
from engine.bsbi import BSBIIndex
from engine.compression import VBEPostings
from engine.letor import Letor

def show_home(request):
    context = {}
    BSBI_instance = BSBIIndex(data_dir='../engine/collections',
                            postings_encoding=VBEPostings,
                            output_dir='../engine/index')
    BSBI_instance.load()
    # letor = Letor()

    if request.method == 'POST':
        queries_str = request.POST.get('queries')

        if queries_str:
            queries = [query.strip() for query in queries_str.split('\n') if query.strip()]
            results = []

            for query in queries:
                query_results = []
                tf_idf_result = BSBI_instance.retrieve_tfidf(query, k=10)
                # tf_idf_result = letor.rerank(query, [t[1] for t in tf_idf_result])
                for (score, doc) in tf_idf_result:
                    did = (re.split(r'[\\/\.]', doc)[-2])
                    query_results.append({'doc': did, 'score': score})

                results.append({'query': query, 'results': query_results})

            context['queries'] = queries
            context['query_results'] = results

            return render(request, 'home.html', context)

    return render(request, 'home.html', context)
