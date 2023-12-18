import os
import re

from django.shortcuts import render, redirect
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from concurrent.futures import ThreadPoolExecutor
from engine.bsbi import BSBIIndex
from engine.compression import VBEPostings
from engine.letor import Letor

def show_home(request):

    if request.method == 'POST':
        queries = request.POST.get('queries')

        return redirect('website:show_page', queries=queries)

    return render(request, 'home.html')

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

def show_page(request, queries):

    if request.method == 'POST':
        queries = request.POST.get('queries')

        return redirect('website:show_page', queries=queries)

    context = {}
    results = []
    query_results = []

    BSBI_instance = BSBIIndex(data_dir='engine/collections',
                            postings_encoding=VBEPostings,
                            output_dir='engine/index')
    BSBI_instance.load()
    letor = Letor()

    tf_idf_result = BSBI_instance.retrieve_tfidf(queries, k=200)
    tf_idf_result = letor.rerank(queries, [t[1] for t in tf_idf_result])
    page_num = request.GET.get('page', 1)
    paginator = Paginator(tf_idf_result, 25)

    try:
        page_obj = paginator.page(page_num)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_page, [(score, doc, queries) for score, doc in page_obj]))

    query_results = [{'doc': result['doc'], 'block': result['block'], 'score': result['score'],
                      'summary': result['summary'], 'title': result['title']} for result in results]

    results.append({'results': query_results})

    context['queries'] = queries
    context['query_results'] = results
    context['page'] = page_obj

    return render(request, 'page.html', context)

def process_page(score_doc_queries):

    score, doc, queries = score_doc_queries

    did = (re.split(r'[\\/\.]', doc)[-2])
    bid = (re.split(r'[\\/\.]', doc)[-3])

    summary_path = os.path.abspath('./engine/collections/' + f'{bid}/{did}.txt')

    with open(summary_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content = ''.join(content)
        content = content.lower()

        summary = ""
        summary_id = content.find(queries)

        if summary_id == -1:
            idx = -1
            for i in range(len(queries.split())):
                idx = content.find(queries.split()[i].lower())
                if idx != -1:
                    break
            
            if idx == -1:
                summary = content[ : 60]
                last = summary.rfind(" ")
                summary = summary[:last]
                
            else:
                i = idx
                if i- 60 < 0:
                    summary = content[: i + len(queries) + 60]
                    last = summary.rfind(" ")
                    summary = summary[:last]
                else:
                    summary = content[ i-60 : i + len(queries) + 60]
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

    return {'doc': did, 'block': bid, 'score': score, 'summary': summary, 'title': title}