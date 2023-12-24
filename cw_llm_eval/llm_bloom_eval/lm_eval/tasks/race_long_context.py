"""
RACE: Large-scale ReAding Comprehension Dataset From Examinations
https://arxiv.org/pdf/1704.04683.pdf

RACE is a large-scale reading comprehension dataset with more than 28,000 passages
and nearly 100,000 questions. The dataset is collected from English examinations
in China, which are designed for middle school and high school students. The dataset
can be served as the training and test sets for machine comprehension.

Homepage: https://www.cs.cmu.edu/~glai1/data/race/
"""
import collections
import datasets
from datasets import load_from_disk
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import os


_CITATION = """
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
"""


class each:
    def __init__(self, f):
        self.f = f

    def __rrshift__(self, other):
        return list(map(self.f, other))

### 900 tokens
noise_text_1 = "Editor's note: In our Behind the Scenes series, CNN correspondents share their experiences in covering news and analyze the stories behind the events. Here, Soledad O'Brien takes users inside a jail where many of the inmates are mentally ill. \
    An inmate housed on the \"forgotten floor,\" where many mentally ill inmates are housed in Miami before trial. MIAMI, Florida (CNN) -- The ninth floor of the Miami-Dade pretrial detention facility is dubbed the \"forgotten floor.\" Here, inmates with the most severe mental illnesses are incarcerated until they're ready to appear in court. Most often, they face drug charges or charges of assaulting an officer --charges that Judge Steven Leifman says are usually \"avoidable felonies.\" He says the arrests often result from confrontations with police. Mentally ill people often won't do what they're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid, delusional, and less likely to follow directions, according to Leifman. \
    So, they end up on the ninth floor severely mentally disturbed, but not getting any real help because they're in jail. We toured the jail with Leifman. He is well known in Miami as an advocate for justice and the mentally ill. Even though we were not exactly welcomed with open arms by the guards, we were given permission to shoot videotape and tour the floor. Go inside the 'forgotten floor' » . At first, it's hard to determine where the people are. The prisoners are wearing sleeveless robes. Imagine cutting holes for arms and feet in a heavy wool sleeping bag -- that's kind of what they look like. They're designed to keep the mentally ill patients from injuring themselves. That's also why they have no shoes, laces or mattresses. \
    Leifman says about one-third of all people in Miami-Dade county jails are mentally ill. So, he says, the sheer volume is overwhelming the system, and the result is what we see on the ninth floor. Of course, it is a jail, so it's not supposed to be warm and comforting, but the lights glare, the cells are tiny and it's loud. We see two, sometimes three men -- sometimes in the robes, sometimes naked, lying or sitting in their cells. \"I am the son of the president. You need to get me out of here!\" one man shouts at me. He is absolutely serious, convinced that help is on the way -- if only he could reach the White House. Leifman tells me that these prisoner-patients will often circulate through the system, occasionally stabilizing in a mental hospital, only to return to jail to face their charges. \
    It's brutally unjust, in his mind, and he has become a strong advocate for changing things in Miami. Over a meal later, we talk about how things got this way for mental patients. Leifman says 200 years ago people were considered \"lunatics\" and they were locked up in jails even if they had no charges against them. They were just considered unfit to be in society. Over the years, he says, there was some public outcry, and the mentally ill were moved out of jails and into hospitals. But Leifman says many of these mental hospitals were so horrible they were shut down. Where did the patients go? Nowhere. The streets. They became, in many cases, the homeless, he says. They never got treatment. \
    Leifman says in 1955 there were more than half a million people in state mental hospitals, and today that number has been reduced 90 percent, and 40,000 to 50,000 people are in mental hospitals. The judge says he's working to change this. Starting in 2008, many inmates who would otherwise have been brought to the \"forgotten floor\" will instead be sent to a new mental health facility -- the first step on a journey toward long-term treatment, not just punishment. Leifman says it's not the complete answer, but it's a start. Leifman says the best part is that it's a win-win solution. The patients win, the families are relieved, and the state saves money by simply not cycling these prisoners through again and again. And, for Leifman, justice is served. E-mail to a friend.\n\n"

# noise_text_1 = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

class RaceLongCtx(Task):
    VERSION = 1
    DATASET_PATH = "race"
    DATASET_NAME = "high"

    cache = {}
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
    np.random.seed(1234)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _collate_data(self, set):
        '''if set in self.cache:
            return self.cache[set]'''
        # One big issue with HF's implementation of this dataset: it makes a
        # separate document for each question; meanwhile, in the GPT3 paper it
        # is shown that one document is made per passage.

        r = collections.defaultdict(list)
        env_path = os.environ.get("HF_DATASETS_CACHE")
        local_name = self.DATASET_PATH
        if os.path.isfile(self.DATASET_PATH):
            local_name = os.path.basename(self.DATASET_PATH)
            local_name = os.path.splitext(local_name)[0]
        local_path = os.path.join(env_path, "disk", local_name)
        if self.DATASET_NAME is not None:
            local_path = os.path.join(local_path, self.DATASET_NAME)
        try:
            process_datas = load_from_disk(local_path)
        except:
            print("load form disk failed:", local_path)
            process_datas = datasets.load_dataset(
                path=self.DATASET_PATH,
                name=self.DATASET_NAME,
            )
            process_datas.save_to_disk(local_path)

        for item in process_datas[set]:
            r[item["article"]].append(item)

        res = list(
            r.values()
            >> each(
                lambda x: {
                    "article": x[0]["article"],
                    "problems": x
                    >> each(
                        lambda y: {
                            "question": y["question"],
                            "answer": y["answer"],
                            "options": y["options"],
                        }
                    ),
                }
            )
        )

        self.cache[set] = res
        #print("lennnnnnnn:", len(res))
        return res

    def training_docs(self):
        return self._collate_data("train")

    def validation_docs(self):
        return self._collate_data("validation")

    def test_docs(self):
        return self._collate_data("test")

    @classmethod
    def get_answer_option(cls, problem):
        answer = cls.letter_to_num[problem["answer"]]
        return problem["options"][answer]

    @classmethod
    def last_problem(cls, doc):
        return doc["problems"][-1]

    def count_token_len(self, text):
        return len(self.adaptor.tok_encode(text))

    def select_noise_text(self, add_noise_num):
        noise_tokens = self.adaptor.tok_encode(noise_text_1)
        noise_seg = self.adaptor._max_length // len(noise_tokens) + 1
        noise_tokens = noise_tokens * noise_seg
        add_noise_tokens = noise_tokens[:add_noise_num]
        # print("add_num:", add_noise_num, " add_noise_tokens:", len(add_noise_tokens))
        return self.adaptor.tok_decode(add_noise_tokens)

    def doc_to_text(self, doc):
        article_str = "Article:" + doc["article"] + "\n\n"
        answer_str = "Question:" + self.last_problem(doc)["question"] + "\nAnswer:" + self.doc_to_target(doc) + "\n\n"
        question_str = ""
        for problem in doc["problems"][:-1]:
            question = "Question:" + problem["question"] + "\n"
            answer = "Answer:" + self.get_answer_option(problem) + "\n"
            question_str += question + answer
        question_str += "Question:" + self.last_problem(doc)["question"] + "\nAnswer:"
        #### 各种tokens长度统计
        #article_len = self.count_token_len(article_str)
        answer_len = self.count_token_len(answer_str)
        #noise_len = self.count_token_len(noise_text_1)
        question_len = self.count_token_len(question_str)

        ### 答案应该所在的tokens位置
        answer_min_offset = self.adaptor.ans_idx * self.adaptor.seq_len
        answer_max_offset = (self.adaptor.ans_idx + 1) * self.adaptor.seq_len
        ### 最少和最多可以加noise_tokens
        add_noise_tokens_min = max(answer_min_offset - question_len, 0)
        add_noise_tokens_max = max(answer_max_offset - question_len - answer_len, 0)
        noise_str = ""
        if add_noise_tokens_max > add_noise_tokens_min:
            add_noise_num = np.random.randint(add_noise_tokens_min, add_noise_tokens_max) ## rand of the segment
            #add_noise_num = add_noise_tokens_min  ## first of the segment
            #print("min:", add_noise_tokens_min, " max:", add_noise_tokens_max, " rand:", add_noise_num, " question_len:", question_len)
            noise_str = self.select_noise_text(add_noise_num)
        ### 合并所有字段
        final_text = article_str + answer_str + noise_str + question_str

        return final_text

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["article"]

    def doc_to_target(self, doc):
        return " " + self.get_answer_option(self.last_problem(doc))

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        problem = self.last_problem(doc)
        ctx = ctx.replace("  ", " ")
        ll_choices = [
            rf.loglikelihood(ctx, " " + problem["options"][i].replace("  ", " "))[0] for i in range(4)
        ]
        return ll_choices

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = self.letter_to_num[self.last_problem(doc)["answer"]]
        pred = np.argmax(results)
        completion_len = np.array([float(len(i)) for i in self.last_problem(doc)["options"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0
        return {
            "acc": int(pred == gold),
            "acc_norm": acc_norm,
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean, 
            "acc_norm": mean,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "acc": True, 
            "acc_norm": True,
        }   

class RaceHighLongCtx(RaceLongCtx):
    VERSION = 1
    DATASET_PATH = "race"
    DATASET_NAME = "high"

class RaceMiddleLongCtx(RaceLongCtx):
    VERSION = 1
    DATASET_PATH = "race"
    DATASET_NAME = "middle"