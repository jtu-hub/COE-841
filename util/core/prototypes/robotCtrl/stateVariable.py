from abc import ABC, abstractmethod

class StateVariable(ABC):
    @staticmethod
    @abstractmethod
    def dim():
        """returns the dimension of the state variable"""
        pass
