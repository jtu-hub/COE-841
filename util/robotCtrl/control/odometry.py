
import numpy as np

from ...core import Angle, Pose, ControlInput

class Odometry(ControlInput):
    def __init__(self, th1: Angle, d: float, th2: Angle, dt: float, coefs = [0.1, 0.1, 0.05, 0.05]):
        super().__init__(dt)
        
        self.th1 = th1
        self.th2 = th2
        self.d = d

        self.coefs = coefs

    def applyControl(self, x0: Pose, motion_noise: bool = False):
        th1_eff = self.th1
        d_eff = self.d
        th2_eff = self.th2

        if motion_noise:
            s_th1  = self.coefs[0] * abs(th1_eff) + self.coefs[1] * d_eff
            s_d    = self.coefs[2] * d_eff        + self.coefs[3] * (abs(th1_eff) + abs(th2_eff))
            s_th2  = self.coefs[0] * abs(th2_eff) + self.coefs[1] * d_eff

            th1_eff = th1_eff + Angle.from_radians(np.random.normal(0, s_th1))
            d_eff = d_eff + np.random.normal(0, s_d)
            th2_eff = th2_eff + Angle.from_radians(np.random.normal(0, s_th2))
                    
        x  = x0.x  + d_eff * (x0.th + th1_eff).cos
        y  = x0.y  + d_eff * (x0.th + th1_eff).sin
        th = x0.th + th1_eff + th2_eff

        return Pose(x, y, th), Odometry(th1_eff, d_eff, th2_eff, self.dt)