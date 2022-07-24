from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import *
import pandas as pd

@csrf_exempt
def main_index(request):
    model_list = get_model()
    code_list = get_code()
    my_list = zip(model_list, code_list)
    data = {
        'my_list': my_list
    }
    return render(request, 'index.html', data)

def main_result(request):
    if request.method == 'POST':
        title = request.POST['title']
        category = request.POST['category']
        content = request.POST['content']
        price = predict_price(title, content)
        data = {
            'title': title,
            'category': category,
            'content': content,
            'price': int(round(price[0][0], -3))
        }
    return render(request, 'result.html', data)
