import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import block_diag

from ..sensors import DetectedFeature
from ..maps import LandmarkMap
from ..control import VelocityControl
from ...core import Pose, Angle, KalmanFilter, plotGaussian

class EkfSlam(KalmanFilter):
    def __init__(self, initial_estimate: np.array, initial_covariance: np.array, n_landmarks: int = 0, dim_mu: int = Pose.dim(), dim_lm: int = DetectedFeature.dim(), q_mat: np.array = np.eye(DetectedFeature.dim()), r_mat: np.array = np.eye(Pose.dim()), gate_threshold: float = 6):
        self.dim_mu = dim_mu
        self.dim_lm = dim_lm
        self.n_lm = n_landmarks

        self.mu = np.vstack([initial_estimate, np.zeros((self.dim_lm * self.n_lm, 1))])
        self.sigma = block_diag(initial_covariance, np.eye(self.dim_lm * self.n_lm))

        self.r_mat = r_mat
        self.q_mat = q_mat

        self.seen_landmarks = []
        self.alpha_pi = gate_threshold

    @property
    def f_mat(self):
        return np.hstack([np.eye(self.dim_mu), np.zeros((self.dim_mu, self.dim_lm * self.n_lm))])

    def f_mat_x(self, correspondence, n_lm: int | None = None):
        n_lm = self.n_lm if n_lm is None else n_lm
        
        n_rows = self.dim_mu + self.dim_lm
        n_cols_left = correspondence * self.dim_lm
        n_cols_right = (n_lm - (correspondence + 1)) * self.dim_lm 
        
        return np.hstack([
            np.vstack([np.eye(self.dim_mu), np.zeros((self.dim_lm, self.dim_mu))]), 
            np.zeros((n_rows, n_cols_left)),
            np.vstack([np.zeros((self.dim_mu, self.dim_lm)), np.eye(self.dim_lm)]),  
            np.zeros((n_rows, n_cols_right))
        ])
    
    def h_mat(self, dx, dy, q, r, c_i, n_lm: int | None = None, known_correspondences: bool = True):
        h_mat = np.array([
            [-r * dx, -r * dy,  0, r * dx, r * dy, 0],
            [   dy  ,   -dx  , -q,  -dy  ,   dx  , 0],
            [   0   ,    0   ,  0,   0   ,    0  , q]
        ])

        h_mat = h_mat @ self.f_mat_x(c_i, n_lm=n_lm)
        
        return h_mat / max(q, 1e-6)

    @property
    def mu_x(self):
        return Pose.from_array(self.mu[0:self.dim_mu])
    
    def mu_lm(self, correspondence, new_value = None):
        if 0 <= correspondence and correspondence < self.n_lm:
            idx_s = self.dim_mu + correspondence * self.dim_lm 
            idx_e = self.dim_mu + (correspondence + 1) * self.dim_lm 

            if new_value is not None:
                self.mu[idx_s:idx_e] = new_value

            return self.mu[idx_s:idx_e]
        elif correspondence == self.n_lm:
            return new_value
        else:
            return None
    
    @property
    def sigma_x(self):
        return self.sigma[:self.dim_mu, :self.dim_mu]
    
    def sigma_lm(self, correspondence):
        n_start = self.dim_mu + correspondence       * self.dim_lm
        n_end   = self.dim_mu + (correspondence + 1) * self.dim_lm
        return self.sigma[n_start:n_end, n_start:n_end]

    def draw(self, ax: plt.Axes, draw_mu: bool = True, color: str | None = None, label: str | None = None, n_sigmas: int = 3, landmark_map: LandmarkMap | None = None, **kwargs):
        plotGaussian(ax, self.mu_x.as_array[:2, :], self.sigma_x[:2, :2], n_sigmas, draw_mu=draw_mu, color=color, label=label, **kwargs)

        for c_i in self.seen_landmarks:
            if landmark_map is not None:
                color_lm = landmark_map.colors[c_i] 
                label_lm = f"{c_i}"
            else:
                color_lm = "#114675"
                label_lm = "lmks"

            plotGaussian(ax, self.mu_lm(c_i)[:2, :], self.sigma_lm(c_i)[:2, :2], n_sigmas, draw_mu=draw_mu, color=color_lm, label=label_lm, **kwargs)

    def predictOdometry(self, u: VelocityControl):
        self.mu[0:self.dim_mu] = u.applyControl(self.mu_x)[0].as_array

        g_mat = block_diag(u.g_mat(self.mu_x), np.eye(self.dim_lm * self.n_lm))
        f_mat = self.f_mat

        self.sigma = g_mat @ self.sigma @ g_mat.T + f_mat.T @ self.r_mat @ f_mat

    def predictKnownCorrespondence(self, detected_landmarks: list[DetectedFeature], correspondences: list[int]):
        for z_i, c_i in zip(detected_landmarks, correspondences):
            mu_x = self.mu_x
            mu_lm = self.mu_lm(c_i)

            dx, dy = 0, 0
            if c_i not in self.seen_landmarks:
                self.seen_landmarks.append(c_i)
                
                dx = z_i.r * (z_i.phi + mu_x.th).cos
                dy = z_i.r * (z_i.phi + mu_x.th).sin
                mu_lm = self.mu_lm(c_i, new_value=np.array([[mu_x.x + dx],[mu_x.y + dy],[z_i.s]]))
            else:
                dx = mu_lm[0,0] - mu_x.x
                dy = mu_lm[1,0] - mu_x.y

            q = max(1e-9, dx ** 2 + dy ** 2)
            r = np.sqrt(q)
            phi = Angle.arctan2(dy, dx) - mu_x.th

            z_hat = DetectedFeature(r, phi, mu_lm[2,0])
            h_mat = self.h_mat(dx, dy, q, r, c_i)

            k_mat = self.sigma @ h_mat.T @ np.linalg.inv(h_mat @ self.sigma @ h_mat.T + self.q_mat)
            i_mat = np.eye(self.dim_mu + self.dim_lm * self.n_lm)
            
            self.mu += k_mat @ (z_i - z_hat).as_array
            self.sigma = (i_mat - k_mat @ h_mat) @ self.sigma

    def predictUnknownCorrespondence(self, detected_landmarks: list[DetectedFeature], eps: float=1e-3):
        for z_i in detected_landmarks:
            pi_min = self.alpha_pi
            c_min = self.n_lm

            mu_x = self.mu_x

            dx_min = z_i.r * (z_i.phi + mu_x.th).cos
            dy_min = z_i.r * (z_i.phi + mu_x.th).sin

            q_min = max(eps, dx_min ** 2 + dy_min ** 2)
            r_min = np.sqrt(q_min)
            phi_min = Angle.arctan2(dy_min, dx_min) - mu_x.th

            mu_lm_new = np.array([[mu_x.x + dx_min],[mu_x.y + dy_min],[z_i.s]])
            
            mu_min = np.vstack([self.mu, mu_lm_new])
            sigma_min = block_diag(self.sigma, np.eye(self.dim_lm))

            z_hat = DetectedFeature(r_min, phi_min, mu_lm_new[2,0])
            h_min = self.h_mat(dx_min, dy_min, q_min, r_min, c_min, n_lm=self.n_lm + 1, known_correspondences=False)

            dz_min = (z_i - z_hat).as_array
            psi_min = h_min @ sigma_min @ h_min.T + self.q_mat

            for c_k in range(self.n_lm):
                mu_lm_k = self.mu_lm(c_k)

                dx_k = mu_lm_k[0,0] - mu_x.x
                dy_k = mu_lm_k[1,0] - mu_x.y

                q_k = max(eps, dx_k ** 2 + dy_k ** 2)
                r_k = np.sqrt(q_k)
                phi_k = Angle.arctan2(dy_k, dx_k) - mu_x.th

                z_hat = DetectedFeature(r_k, phi_k, mu_lm_k[2,0])
                h_mat = self.h_mat(dx_k, dy_k, q_k, r_k, c_k, known_correspondences=False)

                dz = (z_i - z_hat).as_array

                psi_mat = h_mat @ self.sigma @ h_mat.T + self.q_mat
                pi_k = dz.T @ np.linalg.pinv(psi_mat) @ dz
                
                if pi_k[0,0] <= pi_min:
                    pi_min = pi_k[0,0]
                    c_min = c_k
                    dz_min = dz.copy()
                    h_min = h_mat.copy()
                    psi_min = psi_mat.copy()
                    mu_min = self.mu
                    sigma_min = self.sigma

            self.n_lm = max(self.n_lm, c_min + 1)

            k_mat = sigma_min @ h_min.T @ np.linalg.inv(psi_min)
            i_mat = np.eye(self.dim_mu + self.dim_lm * self.n_lm)

            self.mu = mu_min + k_mat @ dz_min
            self.sigma = (i_mat -k_mat @ h_min) @ sigma_min

            if c_min not in self.seen_landmarks:
                self.seen_landmarks.append(int(c_min))
