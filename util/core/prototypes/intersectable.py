from abc import ABC, abstractmethod

class Intersectable(ABC):
    @abstractmethod
    def intersect(self, other: 'Intersectable') -> tuple[any, float]:
        """
        Computes distance from origin of `other` Intersectable to `self`.

        Returns:
          Any: e.g. intersection coordinates
          float: distance `other`->`self`
        """
        pass
