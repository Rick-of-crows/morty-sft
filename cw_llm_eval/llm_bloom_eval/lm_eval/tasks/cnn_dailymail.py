import numpy as np
import datasets
from lm_eval import metrics
#from rouge_score import rouge_scorer, scoring
from lm_eval.base import rf, Task
from lm_eval.metrics import mean

class CNNDailyMail(Task):
    VERSION = 1
    DATASET_PATH = "cnn_dailymail"
    DATASET_NAME = "1.0.0"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        raise NotImplementedError()

    def validation_docs(self):
        return self.dataset['validation']

    def test_docs(self):
        return self.dataset['test']
        #xx = [info for info in self.dataset['test']]
        #return xx[:200]

    def doc_to_text(self, doc):
        #print("doc:", doc['article'])
        return doc['article']

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["article"]

    def doc_to_target(self, doc):
        return doc["highlights"]

    def construct_requests(self, doc, ctx):
        #print("ctx:", ctx)
        return rf.greedy_until(ctx + " TL;DR:", ["\n"])
        #return rf.greedy_until(ctx + " summarize:", ["\n"])

    def process_results(self, doc, results):
        ref = doc["highlights"]
        result = results
        if isinstance(results, list):
            assert len(results) == 1
            result = results[0]
        ref_pred = (ref, result)
        return {"rouge": ref_pred}

    def aggregation(self):
        return {"rouge": metrics.rouge}

    def higher_is_better(self):
        return {"rouge:", True}