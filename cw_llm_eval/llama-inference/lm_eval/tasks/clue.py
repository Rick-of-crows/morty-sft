import numpy as np
from lm_eval.base import MultipleChoiceTask
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, yesno

class C3(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "clue"
    DATASET_NAME = "c3"

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
        #print("validation_docs:", self.dataset["validation"])
        return map(self._process_doc, self.dataset["validation"])

    def get_gold(self, choices, answer):
        gold = 0
        for idx, choice in enumerate(choices):
            if choice == answer:
                gold = idx
        return gold

    def _process_doc(self, doc):
        passage = " ".join(doc['context'])
        #print("xxx:", passage, " yyy:", doc["question"])
        text = passage + "\n问题: " + doc["question"]
        out_doc = {
            "query": text,
            "choices": [cc for cc in doc["choice"]],
            "gold": self.get_gold(doc["choice"], doc["answer"]),
        }
        #print("out_doc:", out_doc)
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

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

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def doc_to_text(self, doc):
        #print("doc:", doc)
        return "{}\nQuestion：{} True, False or Neither?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        answer = ""
        for key in doc:
            val = doc[key]
            label_list = val["label"]
            gold = 0
            for idx, label in enumerate(label_list):
                if label == 0:
                    gold = idx
                    break
            answer = val["span1_text"][gold]
        return " {}".format(answer)

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

class CLUEWSC(Task):
    VERSION = 0
    DATASET_PATH = "clue"
    DATASET_NAME = "cluewsc2020"

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
        result_infos = self.process_validation()
        return result_infos

    def process_validation(self):
        result_infos = []
        valid_docs = list(self.dataset["validation"])
        result_dict = {}
        for doc in valid_docs:
            text = doc["text"]
            #print("doc:", doc)
            if text not in result_dict:
                result_dict[text] = {"label":[doc["label"]], 
                "span1_text":[doc['target']["span1_text"]],
                "span2_text":doc['target']["span2_text"],
                "span2_index":doc['target']["span2_index"],
                }
            else:
                result_dict[text]["label"].append(doc["label"])
                result_dict[text]["span1_text"].append(doc['target']["span1_text"])
        for key in result_dict:
            val = result_dict[key]
            result_infos.append({key:val})
        return result_infos

    def doc_to_text(self, doc):
        return list(doc.keys())[0]
        #return doc

    def doc_to_target(self, doc):
        return " {}".format({0: "True", 1:"False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        val = doc[ctx]
        span1 = val["span1_text"]
        label = val["label"]
        span2 = val["span2_text"]
        span2_index = int(val["span2_index"])
        prefix = ctx[:span2_index]
        target = ctx[span2_index + len(span2):]
        #rep = ctx[span2_index:span2_index + len(span2)]
        lls = []
        for ss in span1:
            partial_ctx = prefix + ss
            #print(partial_ctx, target)
            lls.append(rf.loglikelihood(partial_ctx, target)[0])
        return lls

    def process_results(self, doc, results):
        gold = 0
        for key in doc:
            val = doc[key]
            label_list = val["label"]
            for idx, label in enumerate(label_list):
                if label == 0:
                    gold = idx
                    break

        pred = np.argmax(results)
        #print("res:", results, "pred:", pred, " gold:", gold)
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}