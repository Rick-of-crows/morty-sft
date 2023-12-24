import re
import inspect
from lm_eval.base import PerplexityTask

class WikiZH(PerplexityTask):
    VERSION = 0
    DATASET_PATH = "suolyer/wiki_zh"
    DATASET_NAME = "test"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        long_text = []
        for test in list(self.dataset["test"]):
            #print("length:", len(test["content"]))
            if len(test["content"]) > 1536:
                long_text.append(test["content"])
        return long_text

    def _process_doc(self, doc):
        #print("doc:", len(doc["content"]))
        return doc["content"]

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        return len(doc)