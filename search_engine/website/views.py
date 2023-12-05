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
    letor = Letor()

    if request.method == 'POST':
        queries_str = request.POST.get('queries')

        if queries_str:
            queries = [query.strip() for query in queries_str.split('\n') if query.strip()]
            results = []

            for query in queries:
                query_results = []
                tf_idf_result = BSBI_instance.retrieve_tfidf(query, k=10)
                docs_tfidf = []
                for doc in [x[1] for x in tf_idf_result]:
                    with open(doc, encoding="utf-8") as file:
                        text = file.read()
                    docs_tfidf.append((doc, text))
                scores = letor.predict(docs_tfidf, query)
                sorted_did_scores = letor.evaluate(docs_tfidf, scores)
                
                for (doc, scores) in sorted_did_scores:
                    query_results.append({'doc': doc, 'score': scores})
                    
                # for (score, doc) in tf_idf_result:
                #     query_results.append({'doc': doc, 'score': score})

                results.append({'query': query, 'results': query_results})

            context['queries'] = queries
            context['query_results'] = results

            return render(request, 'home.html', context)

    return render(request, 'home.html', context)
