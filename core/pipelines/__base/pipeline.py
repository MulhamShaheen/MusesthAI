from abc import ABC, abstractmethod


class Pipeline(ABC):
    def __init__(self, config, **kwargs):
        self.config = config

    @abstractmethod
    def invoke(self, inputs, *args, **kwargs):
        pass
