from .metric_base import BaseMetric
from utils import *

class ScoreEval(BaseMetric):
    def __init__(self, num_player):
        self.max_score = 5
        self.players = num_player
        self.init_res_dict()

    def init_res_dict(self):
        total_res = {"max_score":0}
        cn_res = {"max_score":0}
        en_res = {"max_score":0}
        for i in range(self.players):
            player_str = "player"+str(i+1)+"_score"
            total_res[player_str] = 0
            cn_res[player_str] = 0
            en_res[player_str] = 0
        self.res_dict = {"total_res":total_res, "cn_res":cn_res, "en_res":en_res}

    def init_res_dict_from_category(self, category):
        category_res = {"max_score":0}
        for i in range(self.players):
            player_str = "player"+str(i+1)+"_score"
            category_res[player_str] = 0
        self.res_dict[category] = category_res

    def set_res_dict(self, y1, dic):
        assert len(y1) == self.players
        for i in range(self.players):
            player_str = "player"+str(i+1)+"_score"
            score = 0 if y1[i] is None else float(y1[i])
            dic[player_str] += score
        dic["max_score"] += self.max_score

    def process_result(self, y1, data):
        lang_type = data["lang"]
        self.set_res_dict(y1, self.res_dict["total_res"])
        if lang_type == "CN":
            self.set_res_dict(y1, self.res_dict["cn_res"])
        elif lang_type == "EN":
            self.set_res_dict(y1, self.res_dict["en_res"])
        task_type = data["type"]
        if task_type not in self.res_dict:
            self.init_res_dict_from_category(task_type)
        self.set_res_dict(y1, self.res_dict[task_type])

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