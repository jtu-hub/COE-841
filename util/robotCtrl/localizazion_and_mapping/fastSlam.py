import copy

import numpy as np
import matplotlib.pyplot as plt

from ...core import Pose, Angle, Map, Sensor
from ..maps import ParticleMap, LandmarkObservation
from ..sensors import DetectedFeature, CameraSensor
from ..control import Odometry
from .particle import Particle

class FastSlam(Map):
    def __init__(self, n_particles: int, roi: tuple[tuple[float, float], tuple[float, float]], init_pose: Pose | None = None, sensor_cls: Sensor = CameraSensor, q_mat: np.array = np.diag([1.0, 0.1, 1e-6])):
        self.observed_landmarks = []
        self.particles = []
        self.sensor_cls = sensor_cls
        self.q_mat = q_mat

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


    def measurementModel(self, p_pose: Pose, x_lm: float, y_lm: float, signature: int, eps: float = 1e-3):
        dx = x_lm - p_pose.x
        dy = y_lm - p_pose.y

        q = max(eps, dx ** 2 + dy ** 2)
        r = np.sqrt(q)
        phi = Angle.arctan2(dy, dx) - p_pose.th

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
        q_mat = self.q_mat

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
                    
                    _, h_mat = self.measurementModel(p.pose, x_lm, y_lm, z_i.s, eps=eps)
                    
                    h_inv = np.linalg.pinv(h_mat)
                    sigma = h_inv @ q_mat @ h_inv.T

                    lm = p.map.addLandmark(LandmarkObservation(Pose(x_lm, y_lm, Angle(0)), z_i.s, sigma))
                else:
                    z_hat, h_mat = self.measurementModel(p.pose, lm.pose.x, lm.pose.y, lm.signature, eps=eps)

                    psi_mat = h_mat @ lm.sigma @ h_mat.T + q_mat
                    k_mat = lm.sigma @ h_mat.T @ np.linalg.pinv(psi_mat)

                    dz = (z_i - z_hat).as_array

                    lm.pose = Pose.from_array(lm.pose.as_array + k_mat @ dz)
                    lm.sigma = (np.eye(3) - k_mat @ h_mat) @ lm.sigma
                    lm.credebility += 1
                    
                    # compute the likelihood of this observation
                    fact = 1 / np.sqrt(2 * np.pi * np.linalg.det(psi_mat))
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
        
    def draw(self, ax: plt.Axes, draw_all_particles: bool = False, add_legend: bool = False, **kwargs):
        if self.fittest == None:
            self.fittest = self.getFittestParticle()
        
        if draw_all_particles:
            for p in self.particles:
                p.draw(ax, color_mu="#746451")

        self.fittest.draw(ax, draw_map=True, add_legend=add_legend, color_mu="#ff8c00", **kwargs)

    def update(self, u: Odometry, detected_features: list[DetectedFeature], eps: float=1e-3):
        self.updateFromOdometry(u)
        self.updateFromSensorModel(detected_features, eps=eps)
        self.updateFittestParticel()
        self.resampleParticles()

class FastSlam2(FastSlam):
    def __init__(self, n_particles: int, roi: tuple[tuple[float, float], tuple[float, float]], init_pose: Pose | None = None, likelyhood_new_feature: float = 0.5, sensor_cls: Sensor = CameraSensor, q_mat: np.array = np.diag([1.0, 0.1, 1e-6]), r_mat: np.array = np.diag([0.05**2, 0.05**2, (0.02)**2])):
        super().__init__(n_particles, roi, init_pose=init_pose, sensor_cls=sensor_cls, q_mat=q_mat)
        
        self.r_mat = r_mat
        self.likelyhood_new_feature = likelyhood_new_feature
           
    def measurementModel(self, p_pose: Pose, x_lm: float, y_lm: float, signature: int, eps: float = 1e-3):        
        dx = x_lm - p_pose.x
        dy = y_lm - p_pose.y

        q = max(eps, dx ** 2 + dy ** 2)
        r = np.sqrt(q)
        phi = Angle.arctan2(dy, dx) - p_pose.th

        h_mat = np.array([
            [-dx / r, -dy / r,  0],
            [ dy / q, -dx / q, -1],
            [   0   ,    0   ,  0]
        ])

        h_mat_lm = np.array([
            [ dx / r, dy / r, 0],
            [-dy / q, dx / q, 0],
            [  0   ,    0  ,  1]
        ])

        z_hat = DetectedFeature(r, phi, signature)

        return z_hat, h_mat, h_mat_lm

    def updateFromOdometry(self, u: Odometry):
        raise NotImplementedError("This Method is not implemented in FastSlam2\n...use FastSlam2.updateEstimates(...)")

    def updateFromSensorModel(self, detected_features: list[DetectedFeature], eps: float=1e-3):
        raise NotImplementedError("This Method is not implemented in FastSlam2\n...use FastSlam2.updateEstimates(...)")
    
    def updateEstimates(self, u: Odometry, detected_features: list[DetectedFeature], eps: float=1e-3):
        q_mat = self.q_mat

        r_mat = self.r_mat
        r_inv = np.linalg.pinv(r_mat)

        for p in self.particles:
            p.w = 1.0

            for z_i in detected_features:
                s_max = None
                h_max = None
                h_lm_max = None
                psi_inv_max = None
                dz_max = None
                x_max = None
                pi_max = self.likelyhood_new_feature
                   
                for lm in p.map.landmarks:
                    x_hat = u.applyControl(p.pose, motion_noise=False)[0]
                    z_hat, h_mat, h_mat_lm = self.measurementModel(x_hat, lm.pose.x, lm.pose.y, lm.s)

                    psi_mat = h_mat_lm @ lm.sigma @ h_mat_lm.T + q_mat
                    psi_inv = np.linalg.pinv(psi_mat)

                    dz_hat = (z_i - z_hat).as_array
                    
                    sigma_x = np.linalg.pinv(h_mat.T @ psi_inv @ h_mat + r_inv)
                    mu_x = sigma_x @ h_mat.T @ psi_inv @ dz_hat + x_hat.as_array

                    x_sample = Pose.from_array(np.random.multivariate_normal(mu_x.flatten(), sigma_x).reshape((3, 1)))

                    z_sample, _, _ = self.measurementModel(x_sample, lm.pose.x, lm.pose.y, lm.s)
                    dz = (z_i - z_sample).as_array

                    pi_sample = np.exp(- (dz.T @ psi_inv @ dz) / 2) / np.sqrt(2 * np.pi * np.linalg.det(psi_mat))
                    
                    if pi_sample > pi_max:
                        pi_max = pi_sample
                        s_max = lm.s
                        h_max = h_mat
                        h_lm_max = h_mat_lm
                        psi_inv_max = psi_inv
                        dz_max = dz
                        x_max = x_sample

                
                lm = p.map.getLandmarkBySignature(s_max)

                if lm is None:
                    x_k = u.applyControl(p.pose, motion_noise=True)[0]

                    p.pose = x_k

                    dx = z_i.r * (x_k.th + z_i.phi).cos
                    dy = z_i.r * (x_k.th + z_i.phi).sin

                    x_lm = x_k.x + dx
                    y_lm = x_k.y + dy
                    
                    _, _, h_mat_lm = self.measurementModel(x_k, x_lm, y_lm, z_i.s, eps=eps)
                    
                    h_inv = np.linalg.pinv(h_mat_lm)
                    sigma = h_inv.T @ q_mat @ h_inv

                    lm = p.map.addLandmark(LandmarkObservation(Pose(x_lm, y_lm, Angle(0)), z_i.s, sigma))
                    p.w = p.w * pi_max
                else:
                    p.pose = x_max

                    k_mat = lm.sigma @ h_lm_max.T @ psi_inv_max

                    lm.pose = Pose.from_array(lm.pose.as_array + k_mat @ dz_max)
                    lm.sigma = (np.eye(3) - k_mat @ h_lm_max) @ lm.sigma
                    lm.credebility += 1
                    
                    l_mat = h_max @ r_mat @ h_max.T + h_lm_max @ lm.sigma @ h_lm_max.T + q_mat
                    
                    w = np.exp(- (dz_max.T @ np.linalg.pinv(l_mat) @ dz_max) / 2) / np.sqrt(2 * np.pi * np.linalg.det(l_mat))

                    p.w = p.w * float(w[0,0])

                self.discardDubiousFeatures(z_i.s, p)

        #normalize weights
        n = sum([p.w for p in self.particles])
        for p in self.particles:
            p.w = p.w / n
        
        return
    
    def update(self, u: Odometry, detected_features: list[DetectedFeature], eps: float=1e-3):
        self.updateEstimates(u, detected_features, eps=eps)
        self.updateFittestParticel()
        self.resampleParticles()
            