from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import *

@csrf_exempt
def main_index(request):
    return render(request, 'index.html')

def main_result(request):
    if request.method == 'POST':
        title = request.POST['title']
        category = request.POST['category']
        content = request.POST['content']
        price = predict_price(title, content)
        data = {
            'title' : title,
            'category' : category,
            'content' : content,
            'price': round(price[0][0])
        }
    return render(request, 'result.html',data)