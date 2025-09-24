import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ...core import Pose, Map, StateVariable, Drawable
from ..sensors.detectedFeature import DetectedFeature

class Landmark(StateVariable, Drawable):
    def __init__(self, pose : Pose, signature : int):
        self.pose = pose
        self._signature = signature

    @property
    def s(self):
        return self._signature
    
    @property
    def signature(self):
      return self._signature
    
    @staticmethod
    def dim():
        return 4
    
    def __str__(self):
        return f"Landmark: {self.pose}, signature {self.s}"
    
    def draw(self, ax, color: str = 'b', **kwargs):
        ax.scatter(self.pose.x, self.pose.y, color=color, **kwargs)

class LandmarkMap(Map):
    def __init__(self, landmarks: list[Landmark]):
        self.landmarks = landmarks
        self.colors = self.generateUniqueColors(len(landmarks))

    def generateUniqueColors(self, n, seed=None):
        if seed is not None:
            random.seed(seed)

        if n <= 10:
            return [mcolors.to_hex(c) for c in plt.cm.tab10.colors[:n]]
        elif n <= 20:
            return [mcolors.to_hex(c) for c in plt.cm.tab20.colors[:n]]
        else:
            #for larger n, sample from CSS4 colors
            named_colors = list(mcolors.CSS4_COLORS.keys())
            random.shuffle(named_colors)
            return [mcolors.CSS4_COLORS[c] for c in named_colors[:n]]
    
    def draw(self, ax: plt.Axes, correspondences: list[int] | None = None, **kwargs):
        if correspondences is not None:
            for c in correspondences:
                self.landmarks[c].draw(ax, color=self.colors[c], **kwargs)
        else: 
            for (landmark, color) in zip(self.landmarks, self.colors):
                landmark.draw(ax, color=color, **kwargs)

    def matchFeatures(self, detected_feature: DetectedFeature):
        match = None
        min_dist = np.inf

        for landmark in self.landmarks:
            distance = np.linalg.norm(detected_feature.s - landmark.s)
            
            if distance < min_dist:
                min_dist = distance
                match = landmark
        
        return match
 