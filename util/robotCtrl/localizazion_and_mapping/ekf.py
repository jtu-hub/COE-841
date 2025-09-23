import numpy as np
import matplotlib.pyplot as plt

from ..sensors import DetectedFeature
from ..maps import LandmarkMap
from ...core import Pose, ControlInput, Angle, KalmanFilter

class Ekf(KalmanFilter):
    def __init__(self, initial_estimate: Pose, initial_covariance_mat: np.array, q_mat = np.array([[1,0,0],[0,1,0],[0,0,1]]), coefs = [1, 1, 1, 1]):
        self.a1 = coefs[0]
        self.a2 = coefs[1]
        self.a3 = coefs[2]
        self.a4 = coefs[3]

        self.q_mat = q_mat

        self.sigma_mat = initial_covariance_mat
        self.mu = initial_estimate

    def predictOdometry(self, u: ControlInput):
        g_mat = u.g_mat(self.mu)

        v_mat = u.v_mat(self.mu)

        m_mat = np.array([
            [self.a1 * (u.v ** 2) + self.a2 * (u.w ** 2),                      0                     ],
            [                     0                     , self.a3 * (u.v ** 2) + self.a4 * (u.w ** 2)],
        ])

        mu_hat, _ = u.applyControl(self.mu)

        sigma_hat = (g_mat @ self.sigma_mat @ g_mat.T) + (v_mat @ m_mat @ v_mat.T)

        self.mu = mu_hat
        self.sigma_mat = sigma_hat

        return mu_hat, sigma_hat

    def predictKnownCorrespondence(self, detected_landmarks: list[DetectedFeature], correspondences: list[int], landmark_map: LandmarkMap):
        mu_hat = self.mu
        sigma_hat = self.sigma_mat

        p_z = 1
        for z_i, c_i in zip(detected_landmarks, correspondences):
            lmk = landmark_map.landmarks[c_i]

            m_x, m_y, m_s = lmk.pose.x, lmk.pose.y, lmk.s 

            dx = m_x - mu_hat.x
            dy = m_y - mu_hat.y

            q = dx ** 2 + dy ** 2
            r_hat = np.sqrt(q)
            phi_hat = Angle.arctan2(dy, dx) - mu_hat.th

            z_i_hat = DetectedFeature(r_hat, phi_hat, m_s)

            h_mat = np.array([
                [-dx / r_hat, -dy / r_hat,  0],
                [   dy / q  ,   -dx / q  , -1],
                [     0     ,      0     ,  0]
            ])

            s_mat = h_mat @ sigma_hat @ h_mat.T + self.q_mat
            inv_s_mat = np.linalg.inv(s_mat)
            k_mat = sigma_hat @ h_mat.T @ inv_s_mat
            
            dz = (z_i - z_i_hat).as_array

            mu_hat = mu_hat + Pose.from_array(k_mat @ dz)
            sigma_hat = (np.identity(3) - k_mat @ h_mat) @ sigma_hat
            
            p_z *= np.sqrt(np.linalg.det(2*np.pi* s_mat)) * np.exp(-(dz.T @ inv_s_mat @ dz) / 2)


        self.mu = mu_hat
        self.sigma_mat = sigma_hat

        return mu_hat, sigma_hat, p_z

    def predictUnknownCorrespondence(self, detected_landmarks: list[DetectedFeature], landmark_map: LandmarkMap):
        mu_hat = self.mu
        sigma_hat = self.sigma_mat

        for z_i in detected_landmarks:
            p_z_max = 0
            for lmk in landmark_map.landmarks:
                m_x, m_y, m_s = lmk.pose.x, lmk.pose.y, lmk.s 

                dx = m_x - mu_hat.x
                dy = m_y - mu_hat.y

                q = dx ** 2 + dy ** 2
                r_hat = np.sqrt(q)
                phi_hat = Angle.arctan2(dy, dx) - mu_hat.th

                z_i_hat = DetectedFeature(r_hat, phi_hat, m_s)

                h_mat = np.array([
                    [-dx / r_hat, -dy / r_hat,  0],
                    [   dy / q  ,   -dx / q  , -1],
                    [     0     ,      0     ,  0]
                ])

                s_mat = h_mat @ sigma_hat @ h_mat.T + self.q_mat
                inv_s_mat = np.linalg.inv(s_mat)

                dz = (z_i - z_i_hat).as_array
                p_z = np.sqrt(np.linalg.det(2 * np.pi * s_mat)) * np.exp(-(dz.T @ inv_s_mat @ dz) / 2)

                if p_z >= p_z_max:
                    p_z_max = p_z
                    h_mat_argmax = h_mat
                    inv_s_mat_argmax = inv_s_mat
                    z_i_hat_argmax = z_i_hat

            k_mat = sigma_hat @ h_mat_argmax.T @ inv_s_mat_argmax
            
            dz = (z_i - z_i_hat_argmax).as_array

            mu_hat = mu_hat + Pose.from_array(k_mat @ dz)
            sigma_hat = (np.identity(3) - k_mat @ h_mat_argmax) @ sigma_hat
            
        self.mu = mu_hat
        self.sigma_mat = sigma_hat

        return mu_hat, sigma_hat
    
    