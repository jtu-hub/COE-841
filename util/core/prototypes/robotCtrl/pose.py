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
  
  def copy(self):
      return Pose(self.x, self.y, self.th)
  
  @staticmethod
  def random(roi: tuple[tuple[float, float], tuple[float, float]]):
    x_roi = roi[0]
    y_roi = roi[1]

    x = np.random.uniform(x_roi[0], x_roi[1])
    y = np.random.uniform(y_roi[0], y_roi[1])
    th = Angle.from_radians(np.random.uniform(0, 2 * np.pi))

    return Pose(x, y, th)
  
  @staticmethod
  def noisy(pose: 'Pose', std_noise_rot: float = np.pi / 4, std_noise_trans: float = 0.5):
    n_rot = Angle.from_radians(np.random.normal(0, std_noise_rot))
    n_trans_x = np.random.normal(0, std_noise_trans)
    n_trans_y = np.random.normal(0, std_noise_trans)

    return Pose(pose.x + n_trans_x, pose.y + n_trans_y, pose.th + n_rot)
      
  
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
  
  def rt(self, shift_rotation: 'Pose'):
    # Apply rotation
    cos_th = shift_rotation.th.cos
    sin_th = shift_rotation.th.sin
    x_rotated = self.x * cos_th - self.y * sin_th
    y_rotated = self.x * sin_th + self.y * cos_th

    # Apply translation
    x_translated = x_rotated + shift_rotation.x
    y_translated = y_rotated + shift_rotation.y

    # Apply rotation to the orientation
    th_translated = self.th + shift_rotation.th

    return Pose(x_translated, y_translated, th_translated)

  def irt(self, shift_rot: 'Pose'):
    # Apply inverse translation
    x_translated = self.x - shift_rot.x
    y_translated = self.y - shift_rot.y

    # Apply inverse rotation
    cos_th = shift_rot.th.cos
    sin_th = shift_rot.th.sin

    x_rotated = x_translated * cos_th + y_translated * sin_th
    y_rotated = -x_translated * sin_th + y_translated * cos_th

    # Apply inverse rotation to the orientation
    th_rotated = self.th - shift_rot.th

    return Pose(x_rotated, y_rotated, th_rotated)
