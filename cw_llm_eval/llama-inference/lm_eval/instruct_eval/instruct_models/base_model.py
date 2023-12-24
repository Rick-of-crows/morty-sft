import abc
from abc import abstractmethod

class BaseM(abc.ABC):

    @abstractmethod
    def rank_generate(self, x, y1, y2):
        pass

    @abstractmethod
    def score_generate(self, x, y):
        pass

    @abstractmethod
    def base_generate(self, x):
        pass

    @abstractmethod
    def init_chain_rank(self):
        pass

    @abstractmethod
    def init_chain_score(self):
        pass
