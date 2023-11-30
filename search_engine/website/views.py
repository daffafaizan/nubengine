from django.shortcuts import render
from engine.bsbi import BSBIIndex
from engine.compression import VBEPostings

def show_home(request):
    context = {}
    BSBI_instance = BSBIIndex(data_dir='../engine/collections',
                            postings_encoding=VBEPostings,
                            output_dir='../engine/index')
    BSBI_instance.load()

    if request.method == 'POST':
        queries_str = request.POST.get('queries')

        if queries_str:
            queries = [query.strip() for query in queries_str.split('\n') if query.strip()]
            results = []

            for query in queries:
                query_results = []
                for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
                    query_results.append({'doc': doc, 'score': score})

                results.append({'query': query, 'results': query_results})

            context['queries'] = queries
            context['query_results'] = results

            return render(request, 'home.html', context)

    return render(request, 'home.html', context)
