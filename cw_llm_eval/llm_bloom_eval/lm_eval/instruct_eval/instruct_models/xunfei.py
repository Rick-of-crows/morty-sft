from .base_model import BaseM
import json
from urllib.parse import urlparse
from urllib.parse import urlencode
import _thread as thread
import base64
import hmac
import ssl
import datetime
import hashlib
from time import mktime
from wsgiref.handlers import format_date_time

# from websocket import create_connection
import websocket

class XunFei(BaseM):
    def __init__(self, config):
        self.cfg = config
        self.APPID = self.cfg['APPID']
        self.APIKey = self.cfg['APIKey']
        self.APISecret = self.cfg['APISecret']
        self.answer = ''
        
        domain_map = {
            'wss://spark-api.xf-yun.com/v1.1/chat': 'general',
            'wss://spark-api.xf-yun.com/v2.1/chat': 'generalv2'
        }
        self.Spark_url = self.cfg['SparkUrl']
        assert self.Spark_url in domain_map

        self.host = urlparse(self.Spark_url).netloc
        self.path = urlparse(self.Spark_url).path
        self.domain = domain_map[self.Spark_url]

    def on_message(self, ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code == 10013:
            message = data['header']['message']
            self.answer = message
            ws.close()
        elif code != 0:
            print(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            # print(content,end ="")
            self.answer += content
            # print(1)
            if status == 2:
                ws.close()

    def run(self, ws, *args):
        data = json.dumps(self.gen_params(question=ws.question))
        ws.send(data)

    # 收到websocket错误的处理
    def on_error(self, ws, error):
        print("### error:", error)


    # 收到websocket关闭的处理
    def on_close(self, ws,one,two):
        print(" ")


    # 收到websocket连接建立的处理
    def on_open(self, ws):
        thread.start_new_thread(self.run, (ws,))

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url

    def gen_params(self, question):
        """
        通过appid和用户的提问来生成请参数
        """
        data = {
            "header": {
                "app_id": self.APPID,
                "uid": "12345"
            },
            "parameter": {
                "chat": {
                    "domain": self.domain, # 取值为[general,generalv2] 指定访问的领域,general指向V1.5版本 generalv2指向V2版本。注意：不同的取值对应的url也不一样！
                    "temperature": 0.5, # 取值为[0,1],默认为0.5 核采样阈值。用于决定结果随机性，取值越高随机性越强即相同的问题得到的不同答案的可能性越高
                    "max_tokens": 2048, # 取值为[1,4096]，默认为2048, 模型回答的tokens的最大长度
                    "top_k": 4, # 取值为[1，6],默认为4, 从k个候选中随机选择⼀个（⾮等概率）
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "user", "content": question}
                    ]
                }
            }
        }
        return data

    def init_chain_rank(self):
        pass

    def init_chain_score(self):
        pass

    def rank_generate(self, x, y1, y2):
        pass

    def score_generate(self, x, y):
        pass

    def base_generate(self, texts):
        self.answer = ""
        assert len(texts) == 1
        texts = [text['instruction'] for text in texts]
        text = texts[0]
        ### 讯飞的url要根据date时间动态生成。
        url = self.create_url()
        # self.ws = create_connection(self.url)
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(url, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close, on_open=self.on_open)
        ws.question = text
        try:
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        except Exception:
            print("Catch Exception:", text)
            self.answer = ""
        # print("answer:", self.answer)
        return [self.answer]