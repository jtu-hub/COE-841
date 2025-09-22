import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import block_diag, sqrtm

from ..sensors import DetectedFeature, CameraSensor
from ..maps import LandmarkMap
from ..robot import Robot
from ...core import Pose, ControlInput, Angle, KalmanFilter

class Ukf(KalmanFilter):
    def __init__(self, initial_estimate: Pose, initial_covariance_mat: np.array, q_mat = np.array([[1,0],[0,1]]), coefs = [1, 10, 1, 10], meas_noise: float = 1e-9, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1., kappa: float = 1e-3):
        self.a1 = coefs[0]
        self.a2 = coefs[1]
        self.a3 = coefs[2]
        self.a4 = coefs[3]

        self.q_mat = q_mat
        self.meas_noise = meas_noise

        self.gamma = gamma

        self.sigma_mat = initial_covariance_mat
        self.mu = initial_estimate
        self.x_mat = None
        self.sigma_points = []
        self.pz = 1

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def augmentedSigmaMat(self, u: ControlInput):
        m_mat = np.array([
            [self.a1 * (u.v ** 2) + self.a2 * (u.w ** 2),                      0                     ],
            [                     0                     , self.a3 * (u.v ** 2) + self.a4 * (u.w ** 2)],
        ])

        return block_diag(self.sigma_mat, m_mat, self.q_mat)

    def augmentedMu(self, n_augmented: int):
        mu_arr = self.mu.as_array
        r, c = mu_arr.shape

        r_zeros = n_augmented - r

        return  np.vstack((mu_arr, np.zeros((r_zeros, c))))
    
    def augmentedStates(self, u: ControlInput):
        s_aug = self.augmentedSigmaMat(u)

        n = s_aug.shape[0]

        m_aug = self.augmentedMu(n)

        return m_aug, s_aug

    def computeSigmaPoints(self, m_aug: np.array, s_aug: np.array):
        n = s_aug.shape[0]
        ukf_lambda = self.alpha**2 * (n + self.kappa) - n
        scale = np.sqrt(n + ukf_lambda) 

        s_aug = 0.5 * (s_aug + s_aug.T)
        eps = 1e-10
        for _ in range(10):
            try:
                chol = np.linalg.cholesky(s_aug)
                break
            except np.linalg.LinAlgError:
                s_aug = s_aug + eps*np.eye(n)
                eps *= 10
        else:
            raise np.linalg.LinAlgError(f"Covariance not SPD even after jitter for matrix:\n {s_aug}")
        
        scaled = chol * scale 
        
        x_i = [m_aug.copy()]
        for i in range(n):
            col = scaled[:, i].reshape(-1, 1)
            x_i.append(m_aug + col)
            x_i.append(m_aug - col)
        
        self.x_mat = np.hstack(x_i)
            
    def getSigmaPoint(self, idx: int):
        n, _ = self.x_mat.shape

        n_mu, _ = self.mu.as_array.shape
        n_z, _ = self.q_mat.shape
        n_u = n - n_mu - n_z

        x_i = self.x_mat[:,idx]

        x_mu = x_i[:n_mu]
        x_u = x_i[n_mu:n_mu + n_u]
        x_z = x_i[n_mu + n_u:]

        return np.reshape(x_mu, (n_mu, 1)), \
               np.reshape(x_u,  (n_u,  1)), \
               np.reshape(x_z,  (n_z,  1))

    def w_m(self, n_rows: int, i: int):
        ukf_lambda = self.alpha**2 * (n_rows + self.kappa) - n_rows
        if i == 0:
            return ukf_lambda / (n_rows + ukf_lambda)
        else:
            return 1 / (2 * (n_rows + ukf_lambda))

    def w_c(self, n_rows: int, i: int):
        ukf_lambda = self.alpha**2 * (n_rows + self.kappa) - n_rows
        if i == 0:
            return ukf_lambda / (n_rows + ukf_lambda) + (1 - self.alpha**2 + self.beta)
        else:
            return 1 / (2 * (n_rows + ukf_lambda))

    def drawSigmaPoint(self, x_mu: Pose, x_i: Pose, ax: plt.Axes | None = None):
        if ax is not None:
            r = Robot(x_mu, [], None)
            r.draw(ax, linewidth=0.5, linestyle='--', r=0.2, colors=['b', 'b'])
            ax.plot([x_mu.x, x_i.x],[x_mu.y, x_i.y],color="b", linewidth=0.5)
            r.updatePosition(x_i)
            r.draw(ax, linewidth=0.5, linestyle='--', r=0.2, colors=['b', 'b'])

    def predictOdometry(self, u: ControlInput, m: LandmarkMap, ax: plt.Axes | None = None):
        m_aug, s_aug = self.augmentedStates(u)

        self.computeSigmaPoints(m_aug, s_aug)

        n_rows, n_cols = self.x_mat.shape

        u_cls = u.getClass()

        #compute mu_hat and sigma points x after transfer function
        x_avg, y_avg = 0, 0
        sin_th, cos_th = 0, 0
        self.sigma_points = []
        for i in range(n_cols):
            x_mu, x_u, _ = self.getSigmaPoint(i)

            u_i = u + u_cls.from_array(x_u, u.dt)
            x_i, _ = u_i.applyControl(Pose.from_array(x_mu), motion_noise = False)
            
            w_i = self.w_m(n_rows, i)
            sin_th += x_i.th.sin * w_i
            cos_th += x_i.th.cos * w_i
            x_avg += x_i.x * w_i
            y_avg += x_i.y * w_i

            self.sigma_points.append(x_i)

            self.drawSigmaPoint(Pose.from_array(x_mu), x_i, ax=ax)
        
        mu_hat = Pose(x_avg, y_avg, Angle.arctan2(sin_th, cos_th))
        self.mu = mu_hat
        mu_hat = mu_hat.as_array

        self.sigma_mat = np.zeros_like(self.sigma_mat, dtype=np.float64)
        for i, x_i in enumerate(self.sigma_points):
            dx = (x_i.as_array - mu_hat).copy()
            dx[2,0] = float(Angle.normalize(dx[2,0]))   # assuming angle is index 2
            self.sigma_mat += self.w_c(n_rows, i) * (dx @ dx.T)

    def _updateAfterMeasurement(self, mu_hat: np.array, dz: np.array, s_mat: np.array, sigma_xz: np.array):
        inv_s_mat = np.linalg.inv(s_mat)
        
        k_mat = sigma_xz @ inv_s_mat

        mu_hat = mu_hat + k_mat @ dz
        mu_hat[2,0] = float(Angle.normalize(mu_hat[2,0]))

        self.mu = Pose.from_array(mu_hat)
        self.sigma_mat = self.sigma_mat - k_mat @ s_mat @ k_mat.T
        self.p_z = np.sqrt(np.linalg.det(2*np.pi* s_mat)) * np.exp(-(dz.T @ inv_s_mat @ dz) / 2)

    def _computeSigmaMatrices(self, z: list[np.array], n_x: int, n_z: int, n_rows: int, z_hat: np.array, mu_hat: np.array, compute_dx: callable, compute_dz: callable):
        s_mat = np.zeros((n_z, n_z), dtype=np.float64)
        sigma_xz = np.zeros((n_x, n_z), dtype=np.float64)
        for i, (x_i, z_i) in enumerate(zip(self.sigma_points,z)):
            w_c = self.w_c(n_rows, i)

            dz = compute_dz(z_i, z_hat)
            dx = compute_dx(x_i.as_array, mu_hat)

            s_mat += w_c * (dz @ dz.T)
            sigma_xz += w_c * (dx @ dz.T)
    
        r_mat = np.eye(n_z) * self.meas_noise
        s_mat += r_mat

        return s_mat, sigma_xz


    def fullMeasurementCorrection(self, m: LandmarkMap, detected_landmarks: list[DetectedFeature]):
        mu_hat = self.mu.as_array
        
        n_rows, _ = self.x_mat.shape
        
        #compute the measurements at sigma points
        z_dim = self.q_mat.shape[0] 
        n_z = len(detected_landmarks) * z_dim
        z_hat = np.zeros((n_z, 1), dtype=np.float64)

        sin_phi = np.zeros((n_z), dtype=np.float64)
        cos_phi = np.zeros((n_z), dtype=np.float64)
        r_avg = np.zeros((n_z), dtype=np.float64)

        z = []
        for i, x_i in enumerate(self.sigma_points):
            _, _, x_z = self.getSigmaPoint(i)
            z_i = np.zeros(n_z)
            w_i = self.w_m(n_rows, i)

            # extract closest matching landmark to measured signature and 
            # compute the corresponding measure for a given sigma point 
            for j, landmark in enumerate(detected_landmarks):
                match = m.matchFeatures(landmark)

                dx, dy = match.pose.x - x_i.x, match.pose.y - x_i.y

                r = np.sqrt(dx**2 + dy**2) + x_z[0,0]
                phi = (Angle.arctan2(dy, dx) - x_i.th) + Angle.from_radians(x_z[1,0])

                r_avg[j*z_dim] += w_i * r
                sin_phi[j*z_dim + 1] += w_i * phi.sin
                cos_phi[j*z_dim + 1] += w_i * phi.cos

                z_i[j*z_dim : (j+1)*z_dim] = np.array([r, float(phi)]).flatten() 

            z_i = z_i.reshape((n_z, 1))

            z.append(z_i)

        z_hat =  (r_avg + np.arctan2(sin_phi, cos_phi)).reshape((n_z, 1))
        n_x, _ = mu_hat.shape

        #compute s_mat and sigma_xz matrices
        def compute_dx(x, mu):
            dx = (x - mu).copy()
            dx[2, 0] = float(Angle.normalize(dx[2, 0]))
            return dx
        
        def compute_dz(z, z_):
            dz = (z - z_).copy()
            for j in range(len(detected_landmarks)):
                idx_phi = j * z_dim + 1
                dz[idx_phi, 0] = float(Angle.normalize(dz[idx_phi, 0]))

            return dz

        s_mat, sigma_xz = self._computeSigmaMatrices(z, n_x, n_z, n_rows, z_hat, mu_hat, compute_dx, compute_dz)

        #format the actual measurements
        z_true = np.zeros(n_z, dtype=np.float64)
        for j, landmark in enumerate(detected_landmarks):
            z_true[j*z_dim : (j+1)*z_dim] = np.array([landmark.r, float(landmark.phi)])
        
        z_true = z_true.reshape((n_z, 1))

        #update sigma and mu
        dz = compute_dz(z_true, z_hat)
        self._updateAfterMeasurement(mu_hat, dz, s_mat, sigma_xz)


    def averageMeasurmentCorrection(self, m: LandmarkMap, detected_landmarks: list[DetectedFeature]):
        #compute the measurements at sigma points
        def averageMeasurement(landmarks, r0 = 0, phi0 = 0):
            r_avg, sin_avg, cos_avg = r0, Angle(phi0).sin, Angle(phi0).cos
            l = len(landmarks)
            for lm in landmarks:
                r_avg += lm.r / l
                sin_avg += lm.phi.sin
                cos_avg += lm.phi.cos

            return np.array([[r_avg],[float(Angle.arctan2(sin_avg, cos_avg))]])

        n_rows, _ = self.x_mat.shape
        z = []
        r_z_hat = 0
        sin_z_hat, cos_z_hat = 0, 0
        for i, x_i in enumerate(self.sigma_points):
            _, _, x_z = self.getSigmaPoint(i)

            readings, _ = CameraSensor.getReadingAt(x_i, m, detection_noise=False)
            
            z_i = averageMeasurement(readings, r0=x_z[0,0], phi0=x_z[1,0])

            w_i = self.w_m(n_rows, i)
            r_z_hat += w_i * z_i[0,0]
            sin_z_hat += w_i * Angle(z_i[1,0]).sin
            cos_z_hat += w_i * Angle(z_i[1,0]).cos

            z.append(z_i)

        z_hat = np.array([[r_z_hat],[float(Angle.arctan2(sin_z_hat, cos_z_hat))]])
        
        mu_hat = self.mu.as_array
        n_x, _ = mu_hat.shape
        n_z, _ = z_hat.shape

        #compute s_mat and sigma_xz matrices
        def compute_dx(x, mu):
            dx = (x - mu).copy()
            dx[2, 0] = float(Angle.normalize(dx[2, 0]))
            return dx
        
        def compute_dz(z, z_):
            dz = (z - z_).copy()
            dz[1, 0] = float(Angle.normalize(dz[1, 0]))

            return dz

        s_mat, sigma_xz = self._computeSigmaMatrices(z, n_x, n_z, n_rows, z_hat, mu_hat, compute_dx, compute_dz)

        #format the actual measurements
        z_true = averageMeasurement(detected_landmarks)
    
        #update sigma and mu
        dz = compute_dz(z_true, z_hat)        
        self._updateAfterMeasurement(mu_hat, dz, s_mat, sigma_xz)

    def measurementCorrection(self, m: LandmarkMap, detected_landmarks: list[DetectedFeature], variant: str = 'B'):
        if variant == 'A':
            self.averageMeasurmentCorrection(m, detected_landmarks)
        elif variant == 'B':
            self.fullMeasurementCorrection(m, detected_landmarks)
        else:
            raise ValueError(f"Wrong value for measurement correction variants:\n    have {variant} when should be variant: str = 'A' | 'B'")
       

    def ukf(self, u: ControlInput, m: LandmarkMap, detected_landmarks: list[DetectedFeature], ax: plt.Axes | None = None, variant: str = 'A'):
        self.predictOdometry(u, m, ax)
        self.measurementCorrection(m, detected_landmarks, variant=variant)
    

