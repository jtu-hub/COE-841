import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import sqrtm

from ...core import Pose, Angle, Map, Sensor
from ..maps import ParticleMap, Landmark, LandmarkObservation
from ..sensors import DetectedFeature, CameraSensor
from ..control import Odometry
from .particle import Particle

class FastSlam(Map):
    def __init__(self, n_particles: int, roi: tuple[tuple[float, float], tuple[float, float]], init_pose: Pose | None = None, sensor_cls: Sensor = CameraSensor):
        self.observed_landmarks = []
        self.particles = []
        self.sensor_cls = sensor_cls

        self.fittest = None

        if init_pose is None:
            for _ in range(n_particles):
                self.particles.append(Particle(Pose.random(roi),ParticleMap([], roi)))
        else:
            for _ in range(n_particles):
                self.particles.append(Particle(Pose.noisy(init_pose), ParticleMap([], roi)))
   
    def discardDubiousFeatures(self, signature: int, particle: Particle):
        for lm in particle.map.landmarks:
            if lm.s == signature: continue

            if self.sensor_cls.isObjectInDetectorRange(particle.pose, lm, particle.map):
                lm.credebility -= 1

            if lm.credebility < 1: particle.map.removeLandmark(lm)


    def measurementModel(self, p: Particle, x_lm: float, y_lm: float, signature: int, eps: float = 1e-3):
        dx = x_lm - p.pose.x
        dy = y_lm - p.pose.y

        q = max(eps, dx ** 2 + dy ** 2)
        r = np.sqrt(q)
        phi = Angle.arctan2(dy, dx) - p.pose.th

        h_mat = np.array([
            [-dx / r, -dy / r,  0],
            [ dy / q, -dx / q, -1],
            [   0   ,    0   ,  0]
        ])

        z_hat = DetectedFeature(r, phi, signature)

        return z_hat, h_mat

    def updateFromOdometry(self, u: Odometry):
        for p in self.particles:
            p.history.append(p.pose.copy)
            p.pose = u.applyControl(p.pose, motion_noise=True)[0]


    def updateFromSensorModel(self, detected_features: list[DetectedFeature], eps: float=1e-3):
        q_mat = np.array([
            [1.0, 0, 0],
            [0, 0.1, 0],
            [0, 0, 1e-6]
        ])

        for p in self.particles:
            p.w = 1.0
            
            x = p.pose.x
            y = p.pose.y
            th = p.pose.th

            for z_i in detected_features:            
                lm = p.map.getLandmarkBySignature(z_i.s)

                if lm is None:
                    dx = z_i.r * (th + z_i.phi).cos
                    dy = z_i.r * (th + z_i.phi).sin

                    x_lm = x + dx
                    y_lm = y + dy
                    
                    _, h_mat = self.measurementModel(p, x_lm, y_lm, z_i.s, eps=eps)
                    
                    h_inv = np.linalg.pinv(h_mat)
                    sigma = h_inv @ q_mat @ h_inv.T

                    lm = p.map.addLandmark(LandmarkObservation(Pose(x_lm, y_lm, Angle(0)), z_i.s, sigma))
                else:
                    z_hat, h_mat = self.measurementModel(p, lm.pose.x, lm.pose.y, lm.signature, eps=eps)

                    psi_mat = h_mat @ lm.sigma @ h_mat.T + q_mat
                    k_mat = lm.sigma @ h_mat.T @ np.linalg.pinv(psi_mat)

                    dz = (z_i - z_hat).as_array

                    lm.pose = Pose.from_array(lm.pose.as_array + k_mat @ dz)
                    lm.sigma = (np.eye(3) - k_mat @ h_mat) @ lm.sigma
                    lm.credebility += 1
                    
                    # compute the likelihood of this observation
                    fact = 1 / (2 * np.pi * np.sqrt(np.linalg.det(psi_mat)))
                    expo = - (dz.T @ np.linalg.pinv(psi_mat) @ dz) / 2.
                    w = fact * np.exp(expo)
                    
                    p.w = p.w * w

                    self.discardDubiousFeatures(z_i.s, p)

        #normalize weights
        n = sum([p.w for p in self.particles])
        for p in self.particles:
            p.w = p.w / n
        
        return
    
    def resampleParticles(self):
        step = 1.0/len(self.particles)
        u = np.random.uniform(0,step)
        c = self.particles[0].w
        i = 0
        new_particles = []
        for _ in self.particles:

            while u > c:
                i = i + 1
                c = c + self.particles[i].w

            new_particle = copy.deepcopy(self.particles[i])
            new_particle.w = 1.0/len(self.particles)
            new_particles.append(new_particle)
            #increase the threshold
            u = u + step
        
        self.particles = new_particles

    def getFittestParticle(self):
        max_p = self.particles[0]
        
        for p in self.particles:
            if p.w >= max_p.w:
                max_p = p

        return max_p
    
    def updateFittestParticel(self):
        self.fittest = self.getFittestParticle()
        
    def draw(self, ax: plt.Axes, draw_all_particles: bool = False, **kwargs):
        if self.fittest == None:
            self.fittest = self.getFittestParticle()
        
        if draw_all_particles:
            for p in self.particles:
                p.draw(ax, color_mu="#746451")

        self.fittest.draw(ax, draw_map=True, color_mu="#ff8c00", **kwargs)

    def update(self, u: Odometry, detected_features: list[DetectedFeature], eps: float=1e-3):
        self.updateFromOdometry(u)
        self.updateFromSensorModel(detected_features, eps=eps)
        self.updateFittestParticel()
        self.resampleParticles()

        
                