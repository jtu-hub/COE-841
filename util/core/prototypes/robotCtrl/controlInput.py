from abc import ABC, abstractmethod
from .pose import Pose

class ControlInput(ABC):
  def __init__(self, dt: float):
    self.dt = dt
  
  def getClass(self):
        return type(self)

  @abstractmethod
  def applyControl(self, x0: Pose, motion_noise: bool = False) -> tuple[Pose, 'ControlInput']:
      pass