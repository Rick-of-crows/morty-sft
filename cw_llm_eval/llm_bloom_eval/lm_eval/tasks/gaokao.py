from lm_eval.base import Task, rf
from lm_eval.base import MultipleChoiceTask
import os, sys
import json
import re

name_en2zh = {
    "Biology": "生物学",
    "Chemistry": "化学",
    "Chinese_Lang_and_Usage": "中国语言文学",
    "English": "英语",
    "History": "历史",
    "Math_II": "数学",
    "Math_I": "数学",
    "Physics": "物理学",
    "Political_Science": "政治学",
}

def create_all_tasks():
    name_list = name_en2zh.keys()
    return {f"gaokao-{sub}": create_task(sub) for sub in name_list}

def create_task(subject):
    class GaoKao(GeneralGaoKao):
        def __init__(self, adaptor=None):
            super().__init__(subject, adaptor)
    return GaoKao

class GeneralGaoKao(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "gaokao"
    DATASET_NAME = None

    def __init__(self, subject, adaptor):
        self.DATASET_NAME = subject
        self.adaptor = adaptor
        super().__init__(adaptor=adaptor)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        env_path = os.environ.get("HF_DATASETS_CACHE")
        data_path = self.DATASET_PATH
        data_name = self.DATASET_NAME
        test_path = os.path.join(env_path, data_path, "test", data_name + "_MCQs.json")
        self.dataset = {}
        self.dataset["test"] = self.process_json(test_path)

    def extract_choices(self, questions):
        infos = questions.split("A.")
        if len(infos) != 2:
            return None
            # print(questions, " and len:", len(infos))
        question, infos_2 = infos
        infos = infos_2.split("B.")
        if len(infos) != 2:
            # print("BBBB:", infos_2)
            return None
        choice_a, infos_2 = infos
        infos = infos_2.split("C.")
        if len(infos) != 2:
            # print("CCCC:", infos_2)
            return None
        choice_b, infos_2 = infos
        infos = infos_2.split("D.")
        if len(infos) != 2:
            # print("DDDD:", infos_2)
            return None
        choice_c, choice_d = infos
        return question.strip(), choice_a.strip(), \
            choice_b.strip(), choice_c.strip(), choice_d.strip()

    def process_json(self, json_file):
        #print("json_file:", json_file)
        with open(json_file, "r", encoding="utf-8") as f:
            contents = json.load(f)
        ret_list = []
        for idx, row in enumerate(contents["example"]):
            keys = ["A", "B", "C", "D"]
            answer = row["answer"]
            if len(answer) > 1 or len(answer) == 0 or answer[0] not in keys:
                # print("continue:", answer)
                continue
            infos = self.extract_choices(row["question"])
            if infos is None:
                continue
            q, a, b, c, d = infos
            # print("q:", q, " a:", a, " b:", b, " c:", c, " d:", d)
            dd = {}
            dd["question"] = q
            dd["choices"] = [a, b, c, d]
            dd["answer"] = answer[0]
            ret_list.append(dd)
        return ret_list

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        raise NotImplementedError()

    def validation_docs(self):
        raise NotImplementedError()

    def test_docs(self):
        # return self.dataset["test"]
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, keys):
            #prompt = "问题：" + doc["question"] + "\n选项:\n"
            prompt = "以下是关于{}的单项选择题，请选择出正确答案。\n".format(name_en2zh[self.DATASET_NAME])
            prompt += "题目:" + doc["question"] + "\n选项:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt += "答案是:"
            return prompt
        keys = ["A", "B", "C", "D"]
        query = format_example(doc, keys)
        # print("query:", query)
        # doc["choices"] = ["A", "B", "C", "D"]
        return {
            "query": query,
            "choices": doc["choices"],
            "gold": keys.index(doc["answer"])
        }

    def doc_to_text(self, doc):
        return doc["query"]