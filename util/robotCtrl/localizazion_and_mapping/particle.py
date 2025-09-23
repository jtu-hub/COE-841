from ...core import Pose
from ..maps import ParticleMap

class Particle:
    def __init__(self, init_pose: Pose, landmark_map: ParticleMap):
        self.pose = init_pose
        self.history = []
        self.map = landmark_map