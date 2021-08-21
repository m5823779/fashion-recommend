import os
from flask import Flask, jsonify, request,Response, redirect,g,  url_for,render_template,flash,abort, session,make_response,Blueprint
from flask.globals import current_app, g
import requests
import json
import tempfile
import stat
import pandas as pd
# Flask app setup
app = Flask(__name__)

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
# from linebot.models import (
#     MessageEvent, TextMessage, TextSendMessage, ImageMessage, VideoMessage, AudioMessage, FlexSendMessage, QuickReply, QuickReplyButton
# )
from linebot.models import *

ec_data=pd.read_csv('/app/data/ec_data.csv')

def get_product_list(img_path="./test2.jpg"):
    try:
        payload = {"img_path":img_path}
        r = requests.post("http://192.168.91.88:8080/get_rec_product", headers = {'content-type':'application/json'}, data=json.dumps(payload), timeout=5).json()
        app.logger.info("get img: "+img_path)
        return r['rusult']
    except Exception as e:
        app.logger.error("get_product_list: "+str(e))
        return []


def build_recommend_template(product_list):
    # product_list = [
    #     {
    #         "title": "純棉英文字樣T恤",
    #         "price": None,
    #         "special_price": 250.0,
    #         "img_url": "https://net.wimg.tw/files/1/products/346523_400_802E7B289E7B4AB.jpg?v=24472",
    #         "url": "https://www.net-fashion.net/product/346531"
    #     },
    #     {
    #         "title": "STEADAY印字T恤",
    #         "price": None,
    #         "special_price": 233.0,
    #         "img_url": "https://net.wimg.tw/files/1/products/351532_400_804E7B4ABE889B2.jpg?v=24472",
    #         "url": "https://www.net-fashion.net/product/351542"
    #     },
    #     {
    #         "title": "簡約印字竹節棉T恤",
    #         "price": None,
    #         "special_price": 250.0,
    #         "img_url": "https://net.wimg.tw/files/1/products/348967_400_002E799BDE889B2.jpg?v=24472",
    #         "url": "https://www.net-fashion.net/product/348967"
    #     },
    #     {
    #         "title": "迪士尼公主印花T恤",
    #         "price": 299.0,
    #         "special_price": None,
    #         "img_url": "https://net.wimg.tw/files/1/products/346643_400_801E6B7BAE781B0E7B4AB.jpg?v=24472",
    #         "url": "https://www.net-fashion.net/product/346651"
    #     },
    #     {
    #         "title": "Holiday印字T恤",
    #         "price": None,
    #         "special_price": 233.0,
    #         "img_url": "https://net.wimg.tw/files/1/products/349608_400_804E7B4ABE889B2.jpg?v=24472",
    #         "url": "https://www.net-fashion.net/product/349623"
    #     }
    # ]
    template = {
            "type": "carousel",
            "contents": [
            
            ]
        }
    for item in product_list:
        template['contents'].append(
            {
            "type": "bubble",
            "size": "kilo",
                "hero": {
                    "type": "image",
                    "url": str(item.get('img_url','')),
                    "size": "full",
                    "aspectRatio": "20:13",
                    "aspectMode": "cover",
                    "action": {
                        "type": "uri",
                        "label": "Line",
                        "uri": str(item.get('url',''))
                    },
                    "offsetTop": "20px"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "offsetTop": "20px",
                    "contents": [
                    {
                        "type": "text",
                        "text": str(item.get('title','')),
                        "weight": "bold",
                        "size": "xl",
                        "contents": []
                    },
                    {
                        "type": "box",
                        "layout": "baseline",
                        "margin": "md",
                        "contents": [
                            {
                                "type": "icon",
                                "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png",
                                "size": "sm"
                            },
                            {
                                "type": "icon",
                                "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png",
                                "size": "sm"
                            },
                            {
                                "type": "icon",
                                "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png",
                                "size": "sm"
                            },
                            {
                                "type": "icon",
                                "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gold_star_28.png",
                                "size": "sm"
                            },
                            {
                                "type": "icon",
                                "url": "https://scdn.line-apps.com/n/channel_devcenter/img/fx/review_gray_star_28.png",
                                "size": "sm"
                            },
                            {
                                "type": "text",
                                "text": "4.0",
                                "size": "sm",
                                "color": "#999999",
                                "flex": 0,
                                "margin": "md",
                                "contents": []
                            }
                        ]
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "spacing": "sm",
                        "margin": "lg",
                        "contents": [
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                            {
                                "type": "text",
                                "text": "價格",
                                "size": "sm",
                                "color": "#AAAAAA",
                                "flex": 1,
                                "contents": []
                            },
                            {
                                "type": "text",
                                "text": str(item.get('price','')),
                                "size": "sm",
                                "color": "#666666",
                                "flex": 5,
                                "wrap": True,
                                "contents": []
                            }
                            ]
                        }
                        # ,{
                        #     "type": "box",
                        #     "layout": "baseline",
                        #     "spacing": "sm",
                        #     "contents": [
                        #     {
                        #         "type": "text",
                        #         "text": "特價",
                        #         "size": "sm",
                        #         "color": "#AAAAAA",
                        #         "flex": 1,
                        #         "contents": []
                        #     },
                        #     {
                        #         "type": "text",
                        #         "text": str(item.get('special_price','')),
                        #         "size": "sm",
                        #         "color": "#666666",
                        #         "flex": 5,
                        #         "wrap": True,
                        #         "contents": []
                        #     }
                        #     ]
                        # }
                        ]
                    }
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "flex": 0,
                    "spacing": "md",
                    "contents": [
                    
                    {
                        "type": "button",
                        "action": {
                        "type": "uri",
                        "label": "前往",
                        "uri": str(item.get('url',''))
                        },
                        "height": "sm",
                        "style": "primary"
                    }
                    ]
                }
            }
        )
    return template

# Channel Access Token
line_bot_api = LineBotApi("ntXhthpo4BZfhyEJS6RSU6m+9qBPmXs3a8oXbCqBSdu37xPKvJGxgG6/1QhSUO3hwrqUpI0LxA91u80yS5PN55Q3T5powJd1HgpJApvVh3uDHsZhJt3irnVdbvDSMiHfAVsT45FXa2bvgCzdnVJHxgdB04t89/1O/w1cDnyilFU=")
# Channel Secret
handler = WebhookHandler('89ef868d3fd423a018b96f18ec39aab4')
# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

static_tmp_path='/app/static/'

@handler.add(MessageEvent, message=(ImageMessage, VideoMessage, AudioMessage))
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    elif isinstance(event.message, VideoMessage):
        ext = 'mp4'
    elif isinstance(event.message, AudioMessage):
        ext = 'm4a'
    else:
        return
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name
    dist_path = tempfile_path + '.' + ext
    dist_name = os.path.basename(dist_path)
    os.rename(tempfile_path, dist_path)
    os.chmod(dist_path,stat.S_IROTH)
    product_list=get_product_list(img_path="/mnt/sa-nas/Docker/jocelyn.settv-it.com"+dist_path)
    os.remove(dist_path)
    if len(product_list)>0:
        template = build_recommend_template(product_list)
        flex_message = FlexSendMessage(
                    alt_text='找到的產品',
                    contents=template
                )
        line_bot_api.reply_message(
            event.reply_token,flex_message
        )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='抱歉,沒有找到相關的商品'))

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text=event.message.text
    if text=="幫我找":
        _list=['男生上身', '女生上身', '男生下身', '女生下身','洋裝/連身裙']
        message = TextSendMessage(text=str("想要找什麼類別的衣服呢？"),
            quick_reply=QuickReply(
                items=[
                    QuickReplyButton(
                        action=MessageAction(label=str(_list[0]),text="請幫我找 "+str(_list[0]))
                    ),
                    QuickReplyButton(
                        action=MessageAction(label=str(_list[1]),text="請幫我找 "+str(_list[1]))
                    ),
                    QuickReplyButton(
                        action=MessageAction(label=str(_list[2]),text="請幫我找 "+str(_list[2]))
                    ),
                    QuickReplyButton(
                        action=MessageAction(label=str(_list[3]),text="請幫我找 "+str(_list[3]))
                    ),
                    QuickReplyButton(
                        action=MessageAction(label=str(_list[4]),text="請幫我找 "+str(_list[4]))
                    )
                ]
            )
        )
        line_bot_api.reply_message(event.reply_token, message)
    if text.replace("請幫我找 ","") in ['男生上身', '女生上身', '男生下身', '女生下身','洋裝/連身裙']:
        text=text.replace("請幫我找 ","")
        if text=="男生上身":
            product_df=ec_data[(ec_data['category']=='上身類')&(ec_data['gender']=='M')].sample(n=5)[['title','price','special_price','img_url','url']].fillna(0)
        elif text=="男生下身":
            product_df=ec_data[(ec_data['category']=='下身類')&(ec_data['gender']=='M')].sample(n=5)[['title','price','special_price','img_url','url']].fillna(0)
        elif text=="女生上身":
            product_df=ec_data[(ec_data['category']=='上身類')&(ec_data['gender']=='F')].sample(n=5)[['title','price','special_price','img_url','url']].fillna(0)
        elif text=="女生下身":
            product_df=ec_data[(ec_data['category']=='下身類')&(ec_data['gender']=='M')].sample(n=5)[['title','price','special_price','img_url','url']].fillna(0)
        elif text=="洋裝/連身裙":  
            product_df=ec_data[ec_data['category']=='洋裝/連身裙'].sample(n=5)[['title','price','special_price','img_url','url']].fillna(0)

        product_df['price']=(product_df['price']+product_df['special_price']).astype('int')
        product_list=product_df[['title','price','img_url','url']].to_dict('recode')
        if len(product_list)>0:
            template = build_recommend_template(product_list)
            flex_message = FlexSendMessage(
                        alt_text='找到的產品',
                        contents=template
                    )
            line_bot_api.reply_message(
                event.reply_token,flex_message
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='抱歉,沒有找到相關的商品'))

# @handler.add(MessageEvent, message=TextMessage)
# def handle_message(event):
#     line_bot_api.reply_message(
#         event.reply_token,
#         TextSendMessage(text='Hello World!'))

@app.route("/ping")
def ping():
    return 'ping!'

@app.route("/")
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=90, debug=True,ssl_context="adhoc")