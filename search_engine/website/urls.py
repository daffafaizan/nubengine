from django.urls import path
from website.views import show_home, show_docs, show_page

app_name = 'website'

urlpatterns = [
    path('', show_home, name='show_home'),
    path('page/<str:queries>/', show_page, name='show_page'),
    path('result/<str:block_id>/<str:file_name>', show_docs, name='show_docs'),
]