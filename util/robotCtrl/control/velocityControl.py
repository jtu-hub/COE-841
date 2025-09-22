import numpy as np

from ...core import ControlInput, Pose, Angle

class VelocityControl(ControlInput):
  def __init__(self, v : float, w : float, dt : float):
    super().__init__(dt)
    self.v = v
    self.w = w

  def __add__(self, other):
    v = self.v + other.v
    w = self.w + other.w
    dt = (self.dt + other.dt) / 2

    return VelocityControl(v,w,dt)

  def __radd__(self, other):
    return self.__add__(other)
  def __iadd__(self, other):
    return self.__add__(other)
  
  @staticmethod
  def from_array(arr: np.array, dt = None | float):
     if dt is not None:
        return VelocityControl(arr[0][0], arr[1][0], dt)
     else:
        return VelocityControl(arr[0][0], arr[1][0], arr[2][0])
     
  def copy(self):
      return VelocityControl(self.v, self.w, self.dt)
  
  def applyControl(self, x0: Pose, motion_noise: bool = False, std_meas_noise: tuple[float, float] = (0.5, 0.5)) -> tuple[Pose, 'VelocityControl']:
    if motion_noise:
        v_eff = self.v + np.random.normal(0, std_meas_noise[0]) if abs(self.v) > 0. else 0.
        w_eff = self.w + np.random.normal(0, std_meas_noise[1]) if abs(self.w) > 0. else 0.
        u_eff = VelocityControl(
            v_eff,
            w_eff,
            self.dt
        )
    else:
        u_eff = self.copy()

    if np.isclose(u_eff.w, 0.):
        #straight-line motion
        x_new = x0.x + u_eff.v * x0.theta.cos * u_eff.dt
        y_new = x0.y + u_eff.v * x0.theta.sin * u_eff.dt
        theta_new = x0.theta
    else:
        #curved motion (unicycle model)
        theta_new = x0.theta + u_eff.dth
        x_new = x0.x + u_eff.r * ( theta_new.sin - x0.theta.sin)
        y_new = x0.y + u_eff.r * (-theta_new.cos + x0.theta.cos)

    return Pose(x_new, y_new, theta_new), u_eff
  
  @property
  def dth(self):
    return Angle.from_radians(self.w * self.dt)
  
  @property
  def dl(self):
    if np.isclose(self.w, 0.):
        return self.v * self.dt
    
    return self.r * float(self.dth)
  
  @property
  def r(self):
    if np.isclose(self.w, 0.):
        return np.inf
    
    return self.v / self.w
  
  def g_mat(self, x0: Pose):
    """
    Jacobian of transfer function, see applyControl(...) w.r.t. control 
    variables x0
    """
    #best estimate of angle before moovement
    sin_th = x0.th.sin
    cos_th = x0.th.cos
        
    if np.isclose(self.w, 0.):
        return np.array([
            [1, 0, -self.dl * sin_th],
            [0, 1,  self.dl * cos_th],
            [0, 0,          1       ]
        ])
    
    #final angle after control input
    sin_th_f = (x0.th + self.dth).sin
    cos_th_f = (x0.th + self.dth).cos

    d_sin = sin_th_f - sin_th
    d_cos = cos_th_f - cos_th

    return np.array([
        [1, 0, self.r * d_cos],
        [0, 1, self.r * d_sin],
        [0, 0,        1      ]
    ])
  
  def v_mat(self, x0: Pose):
    """
    Jacobian of transfer function, see applyControl(...) w.r.t. control 
    variables x0
    """
    #best estimate of angle before moovement
    sin_th = x0.th.sin
    cos_th = x0.th.cos
    
    if np.isclose(self.w, 0.):
        return np.array([
            [self.dt * cos_th, 0],
            [self.dt * sin_th, 0],
            [        0       , 0]
        ])
    
    #final angle after control input
    sin_th_f = (x0.th + self.dth).sin
    cos_th_f = (x0.th + self.dth).cos

    d_sin = sin_th_f - sin_th
    d_cos = cos_th_f - cos_th

    return np.array([
            [ d_sin / self.w, self.r * (cos_th_f * self.dt - d_sin / self.w)],
            [-d_cos / self.w, self.r * (sin_th_f * self.dt + d_cos / self.w)],
            [      0     ,                       self.dt                    ]
    ])
  
  def __str__(self):
     return f"VelocityControl({self.v}, {self.w}, {self.dt})"

def generateConstantVelocityTrajectory(n_control_inputs: int, v: float, w: float, dt: float):
   return [VelocityControl(v, w, dt) for _ in range(n_control_inputs)]

def generateRandomVelocityTrajectory(n_control_inputs: int, dt: float, max_v: float = 1, max_w: float = 0.5):

    """
    Generate a random, continuous trajectory for `n_control_inputs` steps.
    The trajectory is composed of `n_segments` (1 to `n_control_inputs // 3`),
    where each segment is constant, linearly increasing, or linearly decreasing.
    """
    n_segments = np.random.randint(1, n_control_inputs // 3 + 1)
    segment_lengths = np.random.randint(max(1, n_control_inputs // (n_segments + 1)), 1 + n_control_inputs // n_segments, size=n_segments)
    segment_lengths[-1] += n_control_inputs - np.sum(segment_lengths)  # Ensure total length is correct

    # Initialize trajectories
    v_trajectory = np.zeros(n_control_inputs)
    omega_trajectory = np.zeros(n_control_inputs)

    # Generate random segments for v and omega
    for trajectory in [v_trajectory, omega_trajectory]:
        start_idx = 0
        for i in range(n_segments):
            length = segment_lengths[i]
            end_idx = start_idx + length

            # Randomly choose segment type: 0=constant, 1=linear increasing, 2=linear decreasing
            segment_type = np.random.randint(0, 3)

            # Random start and end values for the segment
            if start_idx == 0:
                start_val = np.random.uniform(-1.0, 1.0)  # Random initial value
            else:
                start_val = trajectory[start_idx - 1]  # Ensure continuity

            if segment_type == 0:  # Constant
                end_val = start_val
            else:  # Linear (increasing or decreasing)
                slope = np.random.uniform(-0.5, 0.5)  # Random slope
                if segment_type == 1:  # Increasing
                    slope = abs(slope)
                else:  # Decreasing
                    slope = -abs(slope)
                end_val = start_val + slope * length

            # Fill the segment
            if length == 1:
                trajectory[start_idx:end_idx] = start_val
            else:
                trajectory[start_idx:end_idx] = np.linspace(start_val, end_val, length)

            start_idx = end_idx

    omega_trajectory -= np.mean(omega_trajectory)
    v_trajectory -= np.mean(v_trajectory)
    us = []
    for v,w in zip(v_trajectory, omega_trajectory):
        if abs(v) >= max_v:
            v = np.sign(v) * max_v

        if abs(w) >= max_w:
            w = np.sign(w) * max_w

        us.append(VelocityControl(v,w,dt))

    return us
