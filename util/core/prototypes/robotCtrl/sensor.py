import numpy as np

from abc import abstractmethod
from ...general.angle import Angle
from ...prototypes.drawable import Drawable
from .map import Map
from .pose import Pose

class Sensor(Drawable):
    def __init__(self, pos: Pose, robot_pose: Pose = Pose(0,0,Angle(0))):
        super().__init__()

        self.rel_pos = pos
        self.robot_pose = robot_pose
        self.reading = None

        self.updatePosition(robot_pose)

    @staticmethod
    @abstractmethod
    def isObjectInDetectorRange(x0: Pose, feature: any):
       pass

    @abstractmethod    
    def getReading(self, m: Map, std_meas_noise: float = 0.5, **kwargs) -> any:
       pass
    
    @property
    def abs_pos(self):
        return Pose(
            self.robot_pose.x + self.rel_pos.x * self.robot_pose.orientation.cos - self.rel_pos.y * self.robot_pose.orientation.sin,
            self.robot_pose.y + self.rel_pos.y * self.robot_pose.orientation.cos + self.rel_pos.x * self.robot_pose.orientation.sin,
            self.robot_pose.orientation + self.rel_pos.orientation
        )

    def updatePosition(self, robot_pose: Pose):
      self.robot_pose = robot_pose