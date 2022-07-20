from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def main_index(request):
    return render(request, 'index.html')

def main_result(request):
    if request.method == 'POST':
        title = request.POST['title']
        category = request.POST['category']
        content = request.POST['content']
        data = {
            'title' : title,
            'category' : category,
            'content' : content
        }
    return render(request, 'result.html',data)
