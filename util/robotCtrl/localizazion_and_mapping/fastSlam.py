import copy

import numpy as np

from ...core import Pose, Angle
from ..maps import ParticleMap, Landmark, LandmarkObservation
from ..sensors import DetectedFeature
from .particle import Particle

class FastSlam:
    def __init__(self, n_particles, m: ParticleMap):
        self.observed_landmarks = []
        self.particles = []

        for _ in range(n_particles):
            self.particles.append(Particle(Pose.random(m.roi),m))

    def isInPerceptualRange(self, landmark: Landmark, particle_pose: Pose):
        #TODO
        return False
    
    def discardDubiousFeatures(self, signature: int, particle: Pose):
        for lm in particle.map.landmarks:
            if lm.s == signature: continue

            if self.isInPerceptualRange(lm, particle.pose):
                lm.credebility -= 1

            if lm.credibility < 1: particle.map.removeLandmark(lm)


    def measurementModel(self, p: Particle, x_lm: float, y_lm: float, signature: int, eps: float = 1e-3):
        dx = x_lm - p.x
        dy = y_lm - p.y

        q = max(eps, dx ** 2 + dy ** 2)
        r = np.sqrt(q)
        phi = Angle.arctan2(dy, dx) - p.th

        h_mat = np.array([
            [-dx / r, -dy / r,  0],
            [ dy / q, -dx / q, -1],
            [   0   ,    0   ,  0]
        ])

        z_hat = DetectedFeature(r, phi, signature)

        return z_hat, h_mat

    def updateFromSensorModel(self, detected_features: list[DetectedFeature], eps: float=1e-3):
        q_mat = np.array(
            [[1.0, 0],
            [0, 0.1]]
        )

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
                    
                    h_inv = np.linalg.inv(h_mat)
                    sigma = h_inv @ q_mat @ h_inv.T

                    lm = p.map.addLandmark(LandmarkObservation(Pose(x_lm, y_lm, Angle(0)), z_i.s, sigma))
                else:
                    z_hat, h_mat = self.measurementModel(p, lm.pose.x, lm.pose.y, lm.signature, eps=eps)

                    psi_mat = h_mat @ lm.sigma @ h_mat.T + q_mat
                    k_mat = lm.sigma @ h_mat.T @ np.linalg.inv(psi_mat)

                    dz = (z_i - z_hat).as_array

                    lm.pose = Pose.from_array(lm.pose.as_array + k_mat @ dz)
                    lm.sigma = (np.eye(2) - k_mat @ h_mat) @ self.sigma
                    lm.credebility += 1
                    
                    # compute the likelihood of this observation
                    fact = 1 / np.sqrt(np.pow(2*np.pi,2) * np.linalg.det(psi_mat))
                    expo = - (dz.T @ np.linalg.inv(psi_mat) @ dz) / 2.
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
        new_particles.append(self.new_particle)
        #increase the threshold
        u = u + step
    
    self.particles = new_particles