import os

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
    letor = Letor()

    if request.method == 'POST':
        queries_str = request.POST.get('queries')

        if queries_str:
            queries = [query.strip() for query in queries_str.split('\n') if query.strip()]
            results = []

            for query in queries:
                query_results = []
                tf_idf_result = BSBI_instance.retrieve_tfidf(query, k=20)
                tf_idf_result = letor.rerank(query, [t[1] for t in tf_idf_result])
                for (score, doc) in tf_idf_result:
                    did = (re.split(r'[\\/\.]', doc)[-2])
                    bid = (re.split(r'[\\/\.]', doc)[-3])

                    summary_path = os.path.abspath('./engine/collections/' + f'{bid}/{did}.txt')

                    with open(summary_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        content = ''.join(content)
                        content = content.lower()
                        summary = ""
                        summary_id = content.find(queries_str)

                        if summary_id == -1:
                            idx = -1
                            for i in range(len(query.split())):
                                idx = content.find(query.split()[i].lower())
                                if idx != -1:
                                    break
                            
                            if idx == -1:
                                summary = content[ : 60]
                                last = summary.rfind(" ")
                                summary = summary[:last]
                                
                            else:
                                i = idx
                                if i- 60 < 0:
                                    summary = content[: i + len(query) + 60]
                                    last = summary.rfind(" ")
                                    summary = summary[:last]
                                else:
                                    summary = content[i-60: i + len(query) + 60]
                                    begin = summary.find(" ")
                                    last = summary.rfind(" ")
                                    summary = summary[begin: last]

                        else:
                            if summary_id - 60 < 0:
                                summary = content[ :summary_id + len(queries) + 60]
                                last = summary.rfind(" ")
                                summary = summary[:last]
                            else:
                                summary = content[summary_id - 60:summary_id + len(queries) + 60]
                                begin = summary.find(" ")
                                last = summary.rfind(" ")
                                summary = summary[begin: last]

                    title_path = os.path.abspath('./engine/generation/title/' + f'{did}.txt')

                    with open(title_path, 'r', encoding='utf-8') as file:
                        title = file.read()

                    query_results.append({'doc': did, 'block': bid, 'score': score, 'summary': summary, 'title': title})

                results.append({'query': query, 'results': query_results,})

            context['queries'] = queries
            context['query_results'] = results



            return render(request, 'home.html', context)

    return render(request, 'home.html', context)

def show_docs(request, block_id, file_name):
    file_path = os.path.abspath('./engine/collections/' + f'{block_id}/{file_name}.txt')

    title_path = os.path.abspath('./engine/generation/title/' + f'{file_name}.txt')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        content_text = ''.join(content)
        content_html = content_text.replace('\n', '<br>')

    with open(title_path, 'r', encoding='utf-8') as file:
        content_title = file.read()

    context = {
        "title": content_title,
        "content": content_html
    }

    return render(request, 'doc.html', context)