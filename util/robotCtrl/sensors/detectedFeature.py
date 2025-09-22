import numpy as np

from ...core import Angle, StateVariable

class DetectedFeature(StateVariable):
  def __init__(self, r : float, phi : Angle, signature : int):
    self.r = r
    self.phi = phi
    self._signature = signature

  @property
  def s(self):
      return self._signature
  
  @property
  def signature(self):
      return self._signature
  
  @property
  def as_array(self):
      return np.array([self.r, float(self.phi), self.s]).reshape((3,1))
  
  @staticmethod
  def dim():
       return 3
  
  @staticmethod
  def from_array(pose_arr: np.array):
      return DetectedFeature(pose_arr[0][0], Angle.from_radians(pose_arr[1][0]), pose_arr[2][0])
  
  def __add__(self, other):
        r = self.r + other.r
        phi = self.phi + other.phi
        s = (self.s + other.s) / 2

        return DetectedFeature(r,phi,s)
  
  def __radd__(self, other):
        return self.__add__(other)
  def __iadd__(self, other):
    return self.__add__(other)
  
  def __truediv__(self, other):
        r = self.r / other
        phi = self.phi / other
        s = self._signature / other

        return DetectedFeature(r,phi,s)
  
  def __mul__(self, other):
        r = self.r * other
        phi = self.phi * other
        s = self._signature * other

        return DetectedFeature(r,phi,s)
  
  def __rmul__(self, other):
        return self.__mul__(other)
  
  def __sub__(self, other):
        r = self.r - other.r
        phi = self.phi - other.phi
        s = self.s - other.s

        return DetectedFeature(r,phi,s)
  
  def __rsub__(self, other):
        r = other.r - self.r
        phi = other.phi - self.phi
        s = other.s - self.s

        return DetectedFeature(r,phi,s)
  
  def __str__(self):
       return f"Detected Feature: ({self.r}, {self.phi}, {self.s})"