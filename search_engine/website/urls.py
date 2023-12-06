from django.urls import path
from website.views import show_home, show_docs

app_name = 'website'

urlpatterns = [
    path('', show_home, name='show_home'),
    path('files/<str:block_id>/<str:file_name>', show_docs, name='show_docs'),
]