from lm_eval.base import Task, rf
from lm_eval.base import MultipleChoiceTask
import pandas as pd
import os, sys
import random

name_en2zh = {
  "computer_network": [
    "Computer Network",
    "计算机网络",
    "STEM"
  ],
  "operating_system": [
    "Operating System",
    "操作系统",
    "STEM"
  ],
  "computer_architecture": [
    "Computer Architecture",
    "计算机组成",
    "STEM"
  ],
  "college_programming": [
    "College Programming",
    "大学编程",
    "STEM"
  ],
  "college_physics": [
    "College Physics",
    "大学物理",
    "STEM"
  ],
  "college_chemistry": [
    "College Chemistry",
    "大学化学",
    "STEM"
  ],
  "advanced_mathematics": [
    "Advanced Mathematics",
    "高等数学",
    "STEM"
  ],
  "probability_and_statistics": [
    "Probability and Statistics",
    "概率统计",
    "STEM"
  ],
  "discrete_mathematics": [
    "Discrete Mathematics",
    "离散数学",
    "STEM"
  ],
  "electrical_engineer": [
    "Electrical Engineer",
    "注册电气工程师",
    "STEM"
  ],
  "metrology_engineer": [
    "Metrology Engineer",
    "注册计量师",
    "STEM"
  ],
  "high_school_mathematics": [
    "High School Mathematics",
    "高中数学",
    "STEM"
  ],
  "high_school_physics": [
    "High School Physics",
    "高中物理",
    "STEM"
  ],
  "high_school_chemistry": [
    "High School Chemistry",
    "高中化学",
    "STEM"
  ],
  "high_school_biology": [
    "High School Biology",
    "高中生物",
    "STEM"
  ],
  "middle_school_mathematics": [
    "Middle School Mathematics",
    "初中数学",
    "STEM"
  ],
  "middle_school_biology": [
    "Middle School Biology",
    "初中生物",
    "STEM"
  ],
  "middle_school_physics": [
    "Middle School Physics",
    "初中物理",
    "STEM"
  ],
  "middle_school_chemistry": [
    "Middle School Chemistry",
    "初中化学",
    "STEM"
  ],
  "veterinary_medicine": [
    "Veterinary Medicine",
    "兽医学",
    "STEM"
  ],
  "college_economics": [
    "College Economics",
    "大学经济学",
    "Social Science"
  ],
  "business_administration": [
    "Business Administration",
    "工商管理",
    "Social Science"
  ],
  "marxism": [
    "Marxism",
    "马克思主义基本原理",
    "Social Science"
  ],
  "mao_zedong_thought": [
    "Mao Zedong Thought",
    "毛泽东思想和中国特色社会主义理论体系概论",
    "Social Science"
  ],
  "education_science": [
    "Education Science",
    "教育学",
    "Social Science"
  ],
  "teacher_qualification": [
    "Teacher Qualification",
    "教师资格",
    "Social Science"
  ],
  "high_school_politics": [
    "High School Politics",
    "高中政治",
    "Social Science"
  ],
  "high_school_geography": [
    "High School Geography",
    "高中地理",
    "Social Science"
  ],
  "middle_school_politics": [
    "Middle School Politics",
    "初中政治",
    "Social Science"
  ],
  "middle_school_geography": [
    "Middle School Geography",
    "初中地理",
    "Social Science"
  ],
  "modern_chinese_history": [
    "Modern Chinese History",
    "近代史纲要",
    "Humanities"
  ],
  "ideological_and_moral_cultivation": [
    "Ideological and Moral Cultivation",
    "思想道德修养与法律基础",
    "Humanities"
  ],
  "logic": [
    "Logic",
    "逻辑学",
    "Humanities"
  ],
  "law": [
    "Law",
    "法学",
    "Humanities"
  ],
  "chinese_language_and_literature": [
    "Chinese Language and Literature",
    "中国语言文学",
    "Humanities"
  ],
  "art_studies": [
    "Art Studies",
    "艺术学",
    "Humanities"
  ],
  "professional_tour_guide": [
    "Professional Tour Guide",
    "导游资格",
    "Humanities"
  ],
  "legal_professional": [
    "Legal Professional",
    "法律职业资格",
    "Humanities"
  ],
  "high_school_chinese": [
    "High School Chinese",
    "高中语文",
    "Humanities"
  ],
  "high_school_history": [
    "High School History",
    "高中历史",
    "Humanities"
  ],
  "middle_school_history": [
    "Middle School History",
    "初中历史",
    "Humanities"
  ],
  "civil_servant": [
    "Civil Servant",
    "公务员",
    "Other"
  ],
  "sports_science": [
    "Sports Science",
    "体育学",
    "Other"
  ],
  "plant_protection": [
    "Plant Protection",
    "植物保护",
    "Other"
  ],
  "basic_medicine": [
    "Basic Medicine",
    "基础医学",
    "Other"
  ],
  "clinical_medicine": [
    "Clinical Medicine",
    "临床医学",
    "Other"
  ],
  "urban_and_rural_planner": [
    "Urban and Rural Planner",
    "注册城乡规划师",
    "Other"
  ],
  "accountant": [
    "Accountant",
    "注册会计师",
    "Other"
  ],
  "fire_engineer": [
    "Fire Engineer",
    "注册消防工程师",
    "Other"
  ],
  "environmental_impact_assessment_engineer": [
    "Environmental Impact Assessment Engineer",
    "环境影响评价工程师",
    "Other"
  ],
  "tax_accountant": [
    "Tax Accountant",
    "税务师",
    "Other"
  ],
  "physician": [
    "Physician",
    "医师资格",
    "Other"
  ]
}

def create_all_tasks():
    name_list = name_en2zh.keys()
    return {f"ceval-{sub}": create_task(sub) for sub in name_list}

def create_task(subject):
    class CEVAL(GeneralCEVAL):
        def __init__(self, adaptor=None):
            super().__init__(subject, adaptor)
    return CEVAL

class GeneralCEVAL(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "ceval"
    DATASET_NAME = None

    def __init__(self, subject, adaptor):
        self.DATASET_NAME = subject
        self.adaptor = adaptor
        super().__init__()

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        env_path = os.environ.get("HF_DATASETS_CACHE")
        data_path = self.DATASET_PATH
        data_name = self.DATASET_NAME
        # local_path = os.path.join(env_path, data_path, data_name)
        dev_path = os.path.join(env_path, data_path, "dev", data_name + "_dev.csv")
        test_path = os.path.join(env_path, data_path, "val", data_name + "_val.csv")
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
            assert len(val) in [6, 7]
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
            prompt = "以下是关于{}的单项选择题，请选择出正确答案。\n".format(name_en2zh[self.DATASET_NAME][1])
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