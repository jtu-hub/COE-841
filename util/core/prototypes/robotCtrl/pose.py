import numpy as np

from ...general.angle import Angle
from ..robotCtrl.stateVariable import StateVariable

class Pose(StateVariable):
  def __init__(self, x : float, y : float, theta : Angle):
    self.x = x
    self.y = y
    self._th = theta

  @property
  def theta(self):
      return self._th
  
  @property
  def orientation(self):
      return self._th
  
  @property
  def th(self):
      return self._th
  
  @property
  def as_array(self):
      return np.array([float(self.x), float(self.y), float(self.th)]).reshape((3,1))
  
  @staticmethod
  def from_array(pose_arr: np.array):
      return Pose(pose_arr[0][0], pose_arr[1][0], Angle.from_radians(pose_arr[2][0]))
  
  @staticmethod
  def dim():
    return 3
  
  def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        th = self.th + other.th

        return Pose(x,y,th)
  
  def __radd__(self, other):
        return self.__add__(other)
  def __iadd__(self, other):
        return self.__add__(other)
  
  def __mul__(self, other):
        x = self.x * other
        y = self.y * other
        th = self.th * other

        return Pose(x,y,th)
  
  def __rmul__(self, other):
        return self.__mul__(other)
  
  def __str__(self):
    return f"({self.x}, {self.y}, {self.orientation})"
