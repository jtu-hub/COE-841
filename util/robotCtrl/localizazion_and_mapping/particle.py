import matplotlib.pyplot as plt

from ...core import Pose, Angle, Drawable
from ..maps import ParticleMap

class Particle(Drawable):
    def __init__(self, init_pose: Pose, landmark_map: ParticleMap):
        self.pose = init_pose
        self.history = []
        self.map = landmark_map
        self.w = 1

    def draw(self, ax: plt.Axes, draw_map: bool = False, color_mu: str = 'b', shift_to: Pose | None = None, **kwargs):
        if shift_to is not None:
            shift_eff = Pose(0, 0, Angle(0)).irt(self.pose).rt(shift_to)
            pose_eff = self.pose.rt(shift_eff)
        else:
            shift_eff = None
            pose_eff = self.pose

        if draw_map: self.map.draw(ax, shift_by=shift_eff,  **kwargs)

        ax.scatter(pose_eff.x, pose_eff.y, marker='x', c=color_mu)