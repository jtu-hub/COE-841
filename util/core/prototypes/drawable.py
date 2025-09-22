import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Drawable(ABC):
    @abstractmethod
    def draw(self, ax: plt.Axes, **kwargs) -> None:
        """
        Draws the object on the provided axes `ax`
        """
        pass