from lm_eval.base import Task, rf
from lm_eval.base import MultipleChoiceTask
import pandas as pd
import os, sys


name_en2zh = {
    "agronomy": "农学",
    "anatomy": "解剖学",
    "ancient_chinese": "古汉语",
    "arts": "艺术学",
    "astronomy": "天文学",
    "business_ethics": "商业伦理",
    "chinese_civil_service_exam": "中国公务员考试",
    "chinese_driving_rule": "中国驾驶规则",
    "chinese_food_culture": "中国饮食文化",
    "chinese_foreign_policy": "中国外交政策",
    "chinese_history":"中国历史",
    "chinese_literature": "中国文学",
    "chinese_teacher_qualification": "中国教师资格",
    "clinical_knowledge": "临床知识",
    "college_actuarial_science":"大学精算学",
    "college_education":"大学教育学",
    "college_engineering_hydrology": "大学工程水文学",
    "college_law": "大学法律",
    "college_mathematics": "大学数学",
    "college_medical_statistics":"大学医学统计",
    "college_medicine": "大学医学",
    "computer_security": "计算机安全",
    "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理",
    "economics": "经济学",
    "education": "教育学",
    "electrical_engineering": "电气工程",
    "elementary_chinese":"小学语文",
    "elementary_commonsense":"小学常识",
    "elementary_information_and_technology": "小学信息技术",
    "elementary_mathematics": "初等数学",
    "ethnology": "民族学",
    "food_science": "食品科学",
    "genetics": "遗传学",
    "global_facts": "全球事实",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "computer_science": "计算机科学",
    "high_school_geography": "高中地理",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理学",
    "high_school_politics": "高中政治",
    "human_sexuality": "人类性行为",
    "international_law": "国际法学",
    "journalism": "新闻学",
    "jurisprudence": "法理学",
    "legal_and_moral_basis": "法律与道德基础",
    "logical": "逻辑学",
    "machine_learning": "机器学习",
    "management": "管理学",
    "marketing": "市场营销",
    "marxist_theory": "马克思主义理论",
    "modern_chinese": "现代汉语",
    "nutrition": "营养学",
    "philosophy": "哲学",
    "professional_accounting": "专业会计",
    "professional_law": "专业法学",
    "professional_medicine": "专业医学",
    "professional_psychology": "专业心理学",
    "public_relations": "公共关系",
    "security_study":"安全研究",
    "sociology": "社会学",
    "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药",
    "virology": "病毒学",
    "world_history":"世界历史",
    "world_religions": "世界宗教",
}

def create_all_tasks():
    name_list = name_en2zh.keys()
    return {f"cmmlu-{sub}": create_task(sub) for sub in name_list}

def create_task(subject):
    class CMMLU(GeneralCMMLU):
        def __init__(self, adaptor=None):
            super().__init__(subject, adaptor)
    return CMMLU

class GeneralCMMLU(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "cmmlu"
    DATASET_NAME = None

    def __init__(self, subject, adaptor):
        self.DATASET_NAME = subject
        self.adaptor = adaptor
        super().__init__(adaptor=adaptor)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        env_path = os.environ.get("HF_DATASETS_CACHE")
        data_path = self.DATASET_PATH
        data_name = self.DATASET_NAME
        # local_path = os.path.join(env_path, data_path, data_name)
        dev_path = os.path.join(env_path, data_path, "dev", data_name + ".csv")
        test_path = os.path.join(env_path, data_path, "test", data_name + ".csv")
        # print("test_path:", test_path)
        self.dataset = {}
        self.dataset["validation"] = self.process_csv(dev_path)
        self.dataset["test"] = self.process_csv(test_path)

    def process_csv(self, csv_path):
        df = pd.read_csv(csv_path, header=0, index_col=0)
        vals = df.values.tolist()
        keys = ["A", "B", "C", "D"]
        doc = {}
        # print("df:", df)
        ret_list = []
        for val in vals:
            assert len(val) == 6
            val = [str(v) for v in val]
            dd = {}
            dd["question"] = val[0]
            dd["choices"] = val[1:5]
            dd["answer"] = val[5]
            ret_list.append(dd)
            # print("question:", val[0], " answer:", val[5])
        return ret_list

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        raise NotImplementedError()

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
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

    def fewshot_examples(self, k, rnd):
        assert k <= 5
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["validation"]))
        
        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

    
