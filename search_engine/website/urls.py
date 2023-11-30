from django.urls import path
from website.views import show_home

app_name = 'website'

urlpatterns = [
    path('', show_home, name='show_home'),
]