# coding=utf-8
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize

# Stance Detection in Chinese Microblogs


class NLPCC(Task):
    VERSION = 0
    DATASET_PATH = "strombergnlp/nlpcc-stance"
    DATASET_NAME = None

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
        return list(self.dataset["train"])
    
    def doc_to_text(self, doc):
        return "{}\nQuestion: Is this sentence about target {} positive or negative?\nAnswer:".format(
            general_detokenize(doc["text"]), doc["target"]
        )

    def doc_to_target(self, doc):
        return " {}".format({1: "FAVOR", 0: "AGAINST", 2: "NONE"}[doc["stance"]])

    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, " FAVOR")
        ll_negative, _ = rf.loglikelihood(ctx, " AGAINST")
        ll_none, _ = rf.loglikelihood(ctx, " NONE")

        return ll_positive, ll_negative, ll_none

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["stance"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}