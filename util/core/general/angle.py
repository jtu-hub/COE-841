import numpy as np

class Angle:
    def __init__(self, degrees: float):
        # store raw radians internally (unwrapped)
        self._val = np.deg2rad(degrees)

    @staticmethod
    def from_radians(radians_value: float) -> "Angle":
        a = Angle(0)
        a._val = float(radians_value)
        return a

    # --- accessors ---
    @property
    def value(self) -> float:
        return self._val

    def __float__(self) -> float:
        return self._val

    @property
    def radians(self) -> float:
        return self._val

    @property
    def degrees(self) -> float:
        return np.rad2deg(self._val)
    
    @property
    def degrees_normalized(self) -> float:
        return np.rad2deg(self.th)

    @property
    def theta(self) -> float:
        val = self._val % (2 * np.pi)

        if val > np.pi:
            val -= 2 * np.pi

        return val

    @property
    def th(self) -> float:
        return self.theta
    
    @property
    def orientation(self) -> float:
        return self.theta

    # --- trig ---
    @property
    def sin(self):
        return np.sin(self._val)

    @property
    def cos(self):
        return np.cos(self._val)

    # --- arithmetic ---
    def __add__(self, other):
        return Angle.from_radians(Angle.from_radians(float(self) + float(other)).th)

    def __sub__(self, other):
        return Angle.from_radians(Angle.from_radians(float(self) - float(other)).th)

    def __mul__(self, other):
        return Angle.from_radians(self._val * other)

    def __truediv__(self, other):
        return Angle.from_radians(float(self) / other)

    def __rtruediv__(self, other):
        return other / float(self)  # returns plain float
    
    def __floordiv__(self, other):
        return Angle.from_radians(self.th / other)
    
    def __rfloordiv__(self, other):
        return other / self.th

    def __neg__(self):
        return Angle.from_radians(-self._val)

    def __repr__(self):
        return f"{self.degrees:.2f}°"
    
    def __abs__(self):
        return abs(float(self))
    
     # --- comparisons ---
    def __eq__(self, other):
        return np.isclose(self.th, other.th, rtol=1e-9, atol=1e-12)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        return self.th < other.th
    
    def __le__(self, other):
        return self.th <= other.th
    
    def __gt__(self, other):
        return self.th > other.th
    
    def __ge__(self, other):
        return self.th >= other.th
    
    def is_between(self, low_bound: 'Angle', high_bound: 'Angle'):
        s = low_bound.th
        e = high_bound.th
        a = self.th

        # shift interval to [0, 2pi]
        s_mod = (s + 2*np.pi) % (2*np.pi)
        e_mod = (e + 2*np.pi) % (2*np.pi)
        a_mod = (a + 2*np.pi) % (2*np.pi)

        if s_mod <= e_mod:
            return s_mod <= a_mod <= e_mod
        else:
            # interval wraps around 2pi
            return a_mod >= s_mod or a_mod <= e_mod

    # --- helpers ---
    @staticmethod
    def arctan2(dy, dx):
        return Angle.from_radians(np.arctan2(dy, dx))
    
    @staticmethod
    def normalize(val: float, return_type: type = float):
        val = val % (2 * np.pi)

        if val > np.pi:
            val -= 2 * np.pi

        if issubclass(return_type, Angle):
            return Angle.from_radians(val)
        else:
            return return_type(val)



