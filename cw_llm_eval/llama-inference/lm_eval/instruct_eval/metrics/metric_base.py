import abc
from abc import abstractmethod
from utils import *

class BaseMetric(abc.ABC):
    def __init__(self):
        self.res_dict = {}

    @abstractmethod
    def process_result(self, result, data):
        pass

    @abstractmethod
    def print_and_save_result(self, path):
        pass