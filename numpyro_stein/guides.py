from abc import ABC, abstractmethod

class ReinitGuide(ABC):
    @abstractmethod
    def init_params(self):
        raise NotImplementedError

    @abstractmethod
    def find_params(self, rng_keys, *args, **kwargs):
        raise NotImplementedError