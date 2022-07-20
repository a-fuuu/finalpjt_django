from importlib.resources import path
from django.urls import URLPattern, path, include

from . import views

urlpatterns = [
    path('', views.main_index, name='index'),
    path('result/', views.main_result, name='result'),
]