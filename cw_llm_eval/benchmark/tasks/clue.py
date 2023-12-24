# coding=utf-8
from functools import partial
from math import exp

import numpy as np

import evaluate
from lm_eval.base import Task, rf
from lm_eval.metrics import f1_score, mean, yesno
from lm_eval.utils import general_detokenize


# sentence pair
class OCNLI(Task):
    VERSION = 0
    DATASET_PATH = "clue"
    DATASET_NAME = "ocnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        # return "在这项任务中，给你一对句子。你的工作是选择这两个句子是否明确一致（蕴含）/不一致（矛盾），或者是无法确定（中立）。你的答案必须是数字0（中性）、1（蕴含）或2（矛盾）的形式。\n句子1：{}\n句子2: {}\n答案:".format(
        #     doc["sentence1"],
        #     doc["sentence2"],
        # )
        return "{}\nQuestion: {} True, False or Neither?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # return " {}".format(doc["label"])

        # True = entailment = 1
        # False = contradiction = 2
        # Neither = neutral = 0
        return " {}".format({1: "True", 0: "Neither", 2: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class CMNLI(Task):
    VERSION = 0
    DATASET_PATH = "clue"
    DATASET_NAME = "cmnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        # return "在这项任务中，给你一对句子。你的工作是选择这两个句子是否明确一致（蕴含）/不一致（矛盾），或者是无法确定（中立）。你的答案必须是数字0（中性）、1（蕴含）或2（矛盾）的形式。\n句子1：{}\n句子2: {}\n答案:".format(
        #     doc["sentence1"],
        #     doc["sentence2"],
        # )
        return "{}\nQuestion: {} True, False or Neither?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # return " {}".format(doc["label"])

        # True = entailment = 1
        # False = contradiction = 2
        # Neither = neutral = 0
        return " {}".format({1: "True", 0: "Neither", 2: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


def yesno_chinese(x):
    if x:
        return "是的"
    else:
        return "不是"


class AFQMC(Task):
    VERSION = 0
    DATASET_PATH = "clue"
    DATASET_NAME = "afqmc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        # return "句子1: {}\n句子2: {}\n问题: 这两个句子的意思是一样的吗?\n答案:".format(
        #     general_detokenize(doc["sentence1"]),
        #     general_detokenize(doc["sentence2"]),
        # )
        return "Question 1: {}\nQuestion 2: {}\nQuestion: Do both questions ask the same thing?\nAnswer:".format(
            general_detokenize(doc["sentence1"]),
            general_detokenize(doc["sentence2"]),
        )

    def doc_to_target(self, doc):
        # return " {}".format(yesno_chinese(doc["label"]))
        return " {}".format(yesno(doc["label"]))

    def construct_requests(self, doc, ctx):
        # ll_yes, _ = rf.loglikelihood(ctx, " 是的")
        # ll_no, _ = rf.loglikelihood(ctx, " 不是")
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


# classify
class TNEWS(Task):
    VERSION = 0
    DATASET_PATH = "clue"
    DATASET_NAME = "tnews"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        # return "将句子'{}'分为下面类别中的一个: news_story, news_culture, news_entertainment, news_sports, news_finance, news_house, news_car, news_edu, news_tech, news_military, news_travel, news_world, news_stock, news_agriculture, news_game\nAnswer:".format(
        #     general_detokenize(doc["sentence"]),
        # )
        return "Divide the following sentence into one of the following categories: news_story, news_culture, news_entertainment, news_sports, news_finance, news_house, news_car, news_edu, news_tech, news_military, news_travel, news_world, news_stock, news_agriculture, news_game.\nSentence:{}\nAnswer:".format(
            general_detokenize(doc["sentence"]),
        )

    def doc_to_target(self, doc):
        label2text = {
            0: "news_story",
            1: "news_culture",
            2: "news_entertainment",
            3: "news_sports",
            4: "news_finance",
            5: "news_house",
            6: "news_car",
            7: "news_edu",
            8: "news_tech",
            9: "news_military",
            10: "news_travel",
            11: "news_world",
            12: "news_stock",
            13: "news_agriculture",
            14: "news_game",
        }

        return " {}".format(label2text[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_1, _ = rf.loglikelihood(ctx, " news_story")
        ll_2, _ = rf.loglikelihood(ctx, " news_culture")
        ll_3, _ = rf.loglikelihood(ctx, " news_entertainment")
        ll_4, _ = rf.loglikelihood(ctx, " news_sports")
        ll_5, _ = rf.loglikelihood(ctx, " news_finance")
        ll_6, _ = rf.loglikelihood(ctx, " news_house")
        ll_7, _ = rf.loglikelihood(ctx, " news_car")
        ll_8, _ = rf.loglikelihood(ctx, " news_edu")
        ll_9, _ = rf.loglikelihood(ctx, " news_tech")
        ll_10, _ = rf.loglikelihood(ctx, " news_military")
        ll_11, _ = rf.loglikelihood(ctx, " news_travel")
        ll_12, _ = rf.loglikelihood(ctx, " news_world")
        ll_13, _ = rf.loglikelihood(ctx, " news_stock")
        ll_14, _ = rf.loglikelihood(ctx, " news_agriculture")
        ll_15, _ = rf.loglikelihood(ctx, " news_game")
        return ll_1, ll_2, ll_3, ll_4, ll_5, ll_6, ll_7, ll_8, ll_9, ll_10, ll_11, ll_12, ll_13, ll_14, ll_15

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


# mrc
# https://github.com/huggingface/evaluate
# https://huggingface.co/spaces/evaluate-metric/squad_v2/blob/main/README.md
# cmrc2018的格式和squad_v2的格式一样
def _squad_metric(predictions, references):
    squad_metric = evaluate.load("./benchmark/metrics/squad_v2/squad_v2.py")
    # pre = ({'id': 'DEV_104_QUERY_0', 'prediction_text': ' 上海轨道交通20号线</s>《海市蜃楼》是一本由陈晓东执导，陈晓东、陈凯歌...', 'no_answer_probability': 3.491428703008022e-11},
    # {'id': 'DEV_609_QUERY_1', 'prediction_text': ' 长沙</s>《海市蜃楼》是一首由歌手陈奕迅演唱的歌曲，由陈奕迅作词...', 'no_answer_probability': 6.846183213793546e-14},
    # {'id': 'DEV_181_QUERY_3', 'prediction_text': ' 虎林（今安徽省贵池市）</s>《海市蜃楼》是一本小说。...', 'no_answer_probability': 3.939706424252185e-13})
    # ref = ({'id': 'DEV_104_QUERY_0', 'answers': {'text': ['上海轨道交通20号线', '上海轨道交通20号线', '上海轨道交通20号线'], 'answer_start': [36, 36, 36]}},
    # {'id': 'DEV_609_QUERY_1', 'answers': {'text': ['沙至益阳', '沙至益阳', '长沙至益阳'], 'answer_start': [105, 105, 104]}},
    # {'id': 'DEV_181_QUERY_3', 'answers': {'text': ['虎林（今安徽省贵池市）', '虎林（今安徽省贵池市）', '虎林'], 'answer_start': [184, 184, 184]}})
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)

    return _squad_metric(predictions=predictions, references=references).get(key, 0)


# EvaluationModule(name: "squad_v2", module_type: "metric", features: {'predictions': {'id': Value(dtype='string', id=None), 'prediction_text': Value(dtype='string', id=None), 'no_answer_probability': Value(dtype='float32', id=None)}, 'references': {'id': Value(dtype='string', id=None), 'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None)}}, usage: """
# Computes SQuAD v2 scores (F1 and EM).
# Args:
#     predictions: List of triple for question-answers to score with the following elements:
#         - the question-answer 'id' field as given in the references (see below)
#         - the text of the answer
#         - the probability that the question has no answer
#     references: List of question-answers dictionaries with the following key-values:
#             - 'id': id of the question-answer pair (see above),
#             - 'answers': a list of Dict {'text': text of the answer as a string}
#     no_answer_threshold: float
#         Probability threshold to decide that a question has no answer.
# Returns:
#     'exact': Exact match (the normalized answer exactly match the gold answer)
#     'f1': The F-score of predicted tokens versus the gold answer
#     'total': Number of score considered
#     'HasAns_exact': Exact match (the normalized answer exactly match the gold answer)
#     'HasAns_f1': The F-score of predicted tokens versus the gold answer
#     'HasAns_total': Number of score considered
#     'NoAns_exact': Exact match (the normalized answer exactly match the gold answer)
#     'NoAns_f1': The F-score of predicted tokens versus the gold answer
#     'NoAns_total': Number of score considered
#     'best_exact': Best exact match (with varying threshold)
#     'best_exact_thresh': No-answer probability threshold associated to the best exact match
#     'best_f1': Best F1 (with varying threshold)
#     'best_f1_thresh': No-answer probability threshold associated to the best F1
# Examples:

#     >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
#     >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
#     >>> squad_v2_metric = evaluate.load("squad_v2")
#     >>> results = squad_v2_metric.compute(predictions=predictions, references=references)
#     >>> print(results)
#     {'exact': 100.0, 'f1': 100.0, 'total': 1, 'HasAns_exact': 100.0, 'HasAns_f1': 100.0, 'HasAns_total': 1, 'best_exact': 100.0, 'best_exact_thresh': 0.0, 'best_f1': 100.0, 'best_f1_thresh': 0.0}
# """, stored examples: 0)


class CMRC2018(Task):
    VERSION = 1
    DATASET_PATH = "clue"
    DATASET_NAME = "cmrc2018"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return f"Context: {doc['context']}\nQuestion: {doc['question']}\nAnswer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = "unanswerable"
        return " " + answer

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
        # import pdb;pdcb.set_trace()
        # 'Context: 本线在2010年前的规划中番号定为上海轨道交通17号线，2011年后改为上海轨道交通20号线...\nQuestion: 上海轨道交通17号线在2010年改名为什么？\nAnswer:'
        continuation = rf.greedy_until(ctx, ["\n"])
        # Req_greedy_until('Context: 本线在2010年前的规划中番号定为上海轨道交通17号线，2011年后改为上海轨道交通20号线...\nQuestion: 上海轨道交通17号线在2010年改名为什么？\nAnswer:', ['\n'])[None]
        is_unanswerable = rf.loglikelihood(ctx, " " + "unanswerable")
        # Req_loglikelihood('context:ninininin \n Question:mimi \n Answer:', 'unanswerable')[None]
        return continuation, is_unanswerable

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation, (logprob_unanswerable, _) = results

        no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
            "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }

        return {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "best_exact": (
                predictions,
                references,
            ),  # Best exact match (with varying threshold)
            "best_f1": (predictions, references),  # Best F1 (with varying threshold)
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "exact": partial(_squad_agg, "exact"),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(_squad_agg, "f1"),  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": partial(
                _squad_agg, "HasAns_exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": partial(_squad_agg, "HasAns_f1"),  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": partial(
                _squad_agg, "NoAns_exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": partial(_squad_agg, "NoAns_f1"),  # The F-score of predicted tokens versus the gold answer
            "best_exact": partial(_squad_agg, "best_exact"),  # Best exact match (with varying threshold)
            "best_f1": partial(_squad_agg, "best_f1"),  # Best F1 (with varying threshold)
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            "best_exact": True,  # Best exact match (with varying threshold)
            "best_f1": True,  # Best F1 (with varying threshold)
        }


class DRCD(Task):
    VERSION = 1
    DATASET_PATH = "clue"
    DATASET_NAME = "drcd"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return f"Context: {doc['context']}\nQuestion: {doc['question']}\nAnswer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = "unanswerable"
        return " " + answer

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
        continuation = rf.greedy_until(ctx, ["\n"])
        is_unanswerable = rf.loglikelihood(ctx, " " + "unanswerable")
        # Req_loglikelihood('context:ninininin \n Question:mimi \n Answer:', 'unanswerable')[None]
        return continuation, is_unanswerable

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation, (logprob_unanswerable, _) = results

        no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
            "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }

        return {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "best_exact": (
                predictions,
                references,
            ),  # Best exact match (with varying threshold)
            "best_f1": (predictions, references),  # Best F1 (with varying threshold)
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "exact": partial(_squad_agg, "exact"),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(_squad_agg, "f1"),  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": partial(
                _squad_agg, "HasAns_exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": partial(_squad_agg, "HasAns_f1"),  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": partial(
                _squad_agg, "NoAns_exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": partial(_squad_agg, "NoAns_f1"),  # The F-score of predicted tokens versus the gold answer
            "best_exact": partial(_squad_agg, "best_exact"),  # Best exact match (with varying threshold)
            "best_f1": partial(_squad_agg, "best_f1"),  # Best F1 (with varying threshold)
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            "best_exact": True,  # Best exact match (with varying threshold)
            "best_f1": True,  # Best F1 (with varying threshold)
        }


# chid是完型填空
# todo chid
