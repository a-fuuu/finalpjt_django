from importlib.resources import path
from django.urls import URLPattern, path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.main_index, name='index'),
    path('result/', views.main_result, name='result'),
] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)