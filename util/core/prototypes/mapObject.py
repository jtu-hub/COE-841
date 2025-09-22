from abc import ABC
from .intersectable import Intersectable
from .drawable import Drawable

class MapObject(Intersectable, Drawable, ABC):
    """
    Any object that populates the map should implement this class
    """
    pass