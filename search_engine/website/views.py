from django.shortcuts import render

def show_home(request):
    context = {}

    if request.method == 'POST':
        query = request.POST.get('query')

        if query is not None:
            context['query'] = query

            return render(request, 'home.html', context)
        
    return render(request, 'home.html')