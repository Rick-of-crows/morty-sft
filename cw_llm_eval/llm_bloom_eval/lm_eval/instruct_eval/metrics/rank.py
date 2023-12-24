from .metric_base import BaseMetric
from utils import *

class RankEval(BaseMetric):
    def __init__(self):
        self.init_res_dict()

    def init_res_dict(self):
        total_res = {"a_win":0, "b_win":0, "tie":0}
        cn_res = {"a_win":0, "b_win":0, "tie":0}
        en_res = {"a_win":0, "b_win":0, "tie":0}
        self.res_dict = {"total_res":total_res, "cn_res":cn_res, "en_res":en_res}

    def init_res_dict_from_category(self, category):
        category_res = {"a_win":0, "b_win":0, "tie":0}
        self.res_dict[category] = category_res

    def set_res_dict(self, result, dic):
        if result == 0:
            dic["a_win"] += 1
        elif result == 1:
            dic["b_win"] += 1
        else:
            dic["tie"] += 1

    def process_result(self, result, data):
        lang_type = data["lang"]
        self.set_res_dict(result, self.res_dict["total_res"])
        if lang_type == "CN":
            self.set_res_dict(result, self.res_dict["cn_res"])
        elif lang_type == "EN":
            self.set_res_dict(result, self.res_dict["en_res"])
        #### cal results by task category
        task_type = data["type"]
        if task_type not in self.res_dict:
            self.init_res_dict_from_category(task_type)
        self.set_res_dict(result, self.res_dict[task_type])

    def print_result(self, dic):
        for key in dic:
            print(key, ":", dic[key])

    def print_and_save_result(self, path):
        print("==================")
        for key in self.res_dict:
            print("------------")
            print(key)
            self.print_result(self.res_dict[key])
        ### save json
        json_dump(path, self.res_dict, ensure_ascii=False)