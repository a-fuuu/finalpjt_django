from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import *
import pandas as pd
import pymysql
import plotly.express as px
import plotly.offline as opy
import time
from sqlalchemy import create_engine
from datetime import datetime, timedelta

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
        conn = pymysql.connect(host = '54.64.211.139',
                        database = 'product_db',
                        user='sg',
                        password='multi1234!')
        cursor = conn.cursor(pymysql.cursors.DictCursor)    
        sql = f'''
        SELECT product_price FROM add_model_table_quantile
        WHERE phone_model_name = '{category}' AND
        product_date > '2022-05';
        '''
        cursor.execute(sql)
        product_data_list = cursor.fetchall()
        df = pd.DataFrame(product_data_list).rename(columns={'product_price':"가격 분포"})
        # price_list = df['product_price']

        price1 = int(round(price[0][0]*1.05, -3))
        price2 = int(round(price[0][0]*0.95, -3))

        hist = px.histogram(
            df['가격 분포'],
            title = f'{category}의 판매가격 분포',
            labels={
                'title':f'{category}의 판매가격',
                'value':'판매 가격대',
                'count':"판매 건수"
                }
            )
        hist.add_scatter(
            y=[0,0],
            x=[price1, price2],
            name="예측 가격",
            marker={'size':15,'symbol':49})
        hist.update_layout(plot_bgcolor='rgb(248, 237, 249)',showlegend=False)
        hist.update_yaxes(title='판매 건수', visible=False)

        graphs1 = opy.plot(
            hist, auto_open=False, output_type='div',
            config={
                'showLink':False,
                'displayModeBar':False,
                'scrollZoom':False})
        db_connection_str = 'mysql+pymysql://jp:multi1234!@54.64.211.139:3306/user_db'
        db_connection = create_engine(db_connection_str, encoding='utf-8')
        current_time = datetime.now() + timedelta(hours=9)
        user_df = pd.DataFrame(data=[[title, content, category, price[0][0], current_time]], columns=['title', 'content','category','price', 'update_time'])
        user_df.to_sql(name='user_input', con=db_connection, if_exists='append', index=False)
        data = {
            'title': title,
            'category': category,
            'content': content,
            # 'price': int(round(price[0][0], -3)),
            'price1': price1,
            'price2': price2,
            'graphs1' : graphs1
        }
    
    return render(request, 'result.html', data)
