"""有道翻译API的调用函数（封装为一个函数使用）"""

import json
import requests
import os, sys, argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                         os.path.pardir)))
from utils import *

class YouDaoTranslator(object):
    def translator(self,str):
        """
        input : str 需要翻译的字符串
        output：translation 翻译后的字符串
        """
        # API
        url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
        # 传输的参数， i为要翻译的内容
        key = {
            'type': "AUTO",
            'i': str,
            "doctype": "json",
            "version": "2.1",
            "keyfrom": "fanyi.web",
            "ue": "UTF-8",
            "action": "FY_BY_CLICKBUTTON",
            "typoResult": "true"
        }
        # key 这个字典为发送给有道词典服务器的内容
        response = requests.post(url, data=key)
        # 判断服务器是否相应成功
        if response.status_code == 200:
            # 通过 json.loads 把返回的结果加载成 json 格式
            result = json.loads(response.text)
            #print ("输入的词为：%s" % result['translateResult'][0][0]['src'])
            #print ("翻译结果为：%s" % result['translateResult'][0][0]['tgt'])
            translation = result['translateResult'][0][0]['tgt']
            return translation
        else:
            print("有道词典调用失败")
            # 相应失败就返回空
            return None

def truthfulqa_trans_yd(args):
    contents = json_load(args.json)
    translator_yd = YouDaoTranslator()
    for idx, content in enumerate(contents):
        #print("idx:", idx, " content:", content)
        contents[idx]['question'] = translator_yd.translator(content['question'])
        ## mc1_targets
        mc1_targets = content['mc1_targets']
        for key in list(mc1_targets.keys()):
            trans_key = translator_yd.translator(key)
            mc1_targets[trans_key] = mc1_targets.pop(key)
        contents[idx]['mc1_targets'] = mc1_targets
        ## mc2_targets
        mc2_targets = content['mc2_targets']
        for key in list(mc2_targets.keys()):
            trans_key = translator_yd.translator(key)
            mc2_targets[trans_key] = mc2_targets.pop(key)
        contents[idx]['mc2_targets'] = mc2_targets
        #print(contents[idx])
    json_dump(args.output, contents, ensure_ascii=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser("translator")
    parser.add_argument("-json", help="input json", required=True)
    parser.add_argument("-output", help="output json", required=True)
    args = parser.parse_args()
    truthfulqa_trans_yd(args)


