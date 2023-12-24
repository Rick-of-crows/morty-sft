from lm_eval.base import Task, rf
from lm_eval.base import MultipleChoiceTask
import os, sys
import json

name_en2zh = {
    "gaokao-biology": "生物学",
    "gaokao-chemistry": "化学",
    "gaokao-chinese": "中国语言文学",
    "gaokao-english": "英语",
    "gaokao-geography": "地理",
    "gaokao-history": "历史",
    "gaokao-mathqa": "数学",
    "gaokao-physics": "物理",
    "jec-qa-ca": "司法考试",
    "jec-qa-kd": "司法考试",
    "logiqa-en": "公务员考试",
    "logiqa-zh": "公务员考试",
    "sat-en": "英语",
    "sat-math": "数学",
}

def create_all_tasks():
    name_list = name_en2zh.keys()
    return {f"AGIEval-{sub}": create_task(sub) for sub in name_list}

def create_task(subject):
    class AGIEval(GeneralAGIEval):
        def __init__(self, adaptor=None):
            super().__init__(subject, adaptor)
    return AGIEval

class GeneralAGIEval(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "AGIEval"
    DATASET_NAME = None

    def __init__(self, subject, adaptor):
        self.DATASET_NAME = subject
        self.adaptor = adaptor
        super().__init__(adaptor=adaptor)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        env_path = os.environ.get("HF_DATASETS_CACHE")
        data_path = self.DATASET_PATH
        data_name = self.DATASET_NAME
        test_path = os.path.join(env_path, data_path, "test", data_name + ".jsonl")
        self.dataset = {}
        self.dataset["test"] = self.process_json(test_path)

    def process_json(self, json_file):
        # print("json_file:", json_file)
        ret_list = []
        with open(json_file, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                answer = data["label"]
                keys = ["A", "B", "C", "D"]
                if len(answer) > 1 or len(answer) == 0 or answer[0] not in keys:
                    # print("continue:", answer)
                    continue
                options = data["options"]
                if len(options) != 4:
                    # print("options:", options)
                    continue
                passage = data["passage"]
                question = data["question"]
                if passage is not None:
                    question = passage + question
                dd = {}
                dd["question"] = question
                dd["choices"] = options
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