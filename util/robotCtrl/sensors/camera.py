import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Arc

from ...core import Angle, Sensor, Pose, Map
from ..maps import Landmark, LandmarkMap
from .detectedFeature import DetectedFeature

class CameraSensor(Sensor):
    def __init__(self, pos: Pose, field_of_view: Angle, robot_pose: Pose = Pose(0,0,Angle(0)), range = 10):
        super().__init__(pos, robot_pose)

        self.fov = field_of_view
        self.range = range
        

    def detectLandmark(self, landmark: Landmark, detection_noise: bool= True, std_meas_noise: float = 0.2, **kwargs) -> tuple[bool, float]:
        dx, dy = landmark.pose.x - self.abs_pos.x, landmark.pose.y - self.abs_pos.y

        r = np.sqrt(dx**2 + dy**2)
        phi = Angle.arctan2(dy, dx) - self.abs_pos.th

        right = -(self.fov / 2)
        left  =  (self.fov / 2)

        if r < self.range and phi.is_between(right, left):
            d_phi1 = abs(phi - right)
            d_phi2 = abs(phi - left)
            
            p_detect = 1 - 0.25 * (d_phi1 + d_phi2) / float(self.fov)
            
            #probability of detection: 1 at center, linear fall off
            is_detected = not detection_noise or np.random.random() < p_detect 
        else:
            is_detected = False

        r_noisy = r + np.random.normal(0, std_meas_noise) if is_detected else None
        phi_noisy = phi + Angle.from_radians(np.random.normal(0, std_meas_noise) / np.pi) if is_detected else None

        return (is_detected, r_noisy, phi_noisy) if detection_noise else (is_detected, r, phi)

    def getReading(self, m: Map, std_meas_noise: float = 0.5, **kwargs) -> any:
        if isinstance(m, LandmarkMap):
            detected_landmarks = []
            correspondences = []
            for c in range(len(m.landmarks)):
                l = m.landmarks[c]

                is_in_fov, r, phi = self.detectLandmark(l, std_meas_noise=std_meas_noise, **kwargs)
            
                if is_in_fov:
                    correspondences.append(c)
                    detected_landmarks.append(
                        self.readingToRobotFrame(DetectedFeature(r, phi, l.s))
                    )
        
            self.reading = (detected_landmarks, correspondences)

            return self.reading
        
        self.reading = None
        return None
    
    @staticmethod
    def getReadingAt(x0:Pose, m: Map, fov: Angle = Angle(60), sensor_range: float = 5, detection_noise: bool= True, std_meas_noise: float = 0.5, **kwargs) -> any:
        if isinstance(m, LandmarkMap):
            detected_landmarks = []
            correspondences = []
            for c in range(len(m.landmarks)):
                l = m.landmarks[c]

                dx, dy = l.pose.x - x0.x, l.pose.y - x0.y

                r = np.sqrt(dx**2 + dy**2)
                phi = Angle.arctan2(dy, dx) - x0.th

                right = -(fov / 2)
                left  =  (fov / 2)
                if r < sensor_range and phi.is_between(right, left):
                    d_phi1 = abs(phi - right)
                    d_phi2 = abs(phi - left)
                    
                    p_detect = 1 - 0.25 * (d_phi1 + d_phi2) / float(fov)

                    #probability of detection: 1 at center, linear fall off
                    is_detected = not detection_noise or np.random.random() < p_detect 
                else:
                    is_detected = False

            
                if is_detected:
                    phi_noisy = phi + Angle.from_radians(np.random.normal(0, std_meas_noise))
                    r_noisy = r + np.random.normal(0, std_meas_noise)
                    
                    correspondences.append(c)
                    detected_landmarks.append(
                        DetectedFeature(r_noisy, phi_noisy, l.s) if detection_noise else DetectedFeature(r, phi, l.s)
                    )
        

            return (detected_landmarks, correspondences)
        
        return None
    
    def readingToRobotFrame(self, reading: DetectedFeature):
        x_s, y_s, th_s = self.abs_pos.x, self.abs_pos.y, self.abs_pos.th
        x_r, y_r, th_r = self.robot_pose.x, self.robot_pose.y, self.robot_pose.th
        r_sf, phi_sf = reading.r, reading.phi

        x_f = x_s + r_sf * (th_s + phi_sf).cos
        y_f = y_s + r_sf * (th_s + phi_sf).sin

        dx = x_f - x_r
        dy = y_f - y_r
        r_rf = np.hypot(dx, dy)
        phi_rf = Angle.arctan2(dy, dx) - th_r

        return DetectedFeature(r_rf, phi_rf, reading.s)

    def draw(self, ax: plt.Axes, color='g', linestyle='--', draw_sensor_readings: bool = False, **kwargs) -> None:
        x, y, theta = self.abs_pos.x, self.abs_pos.y, self.abs_pos.th
        half_fov = self.fov // 2

        #calculate endpoints of the cone sides
        left_angle = theta + half_fov
        right_angle = theta - half_fov

        x_left = x + self.range * left_angle.cos
        y_left = y + self.range * left_angle.sin

        x_right = x + self.range * right_angle.cos
        y_right = y + self.range * right_angle.sin

        #draw the cone as a triangle
        ax.plot([x, x_left], [y, y_left], linestyle=linestyle, color=color, **kwargs)
        ax.plot([x, x_right], [y, y_right], linestyle=linestyle, color=color, **kwargs)

        if draw_sensor_readings and self.reading is not None:
            x_r, y_r, th_r = self.robot_pose.x, self.robot_pose.y, self.robot_pose.th
            for f in self.reading[0]:
                x_f = x_r + f.r * (f.phi + th_r).cos
                y_f = y_r + f.r * (f.phi + th_r).sin
                
                ax.plot([self.abs_pos.x, x_f],[self.abs_pos.y, y_f], linestyle='--', color='b')

        if self.range == np.inf: return

        #draw the circular arc at the bottom
        arc = Arc(
            (x, y),  # Center of the arc
            2 * self.range,  # Width
            2 * self.range,  # Height
            theta1=(right_angle).degrees,
            theta2=(left_angle).degrees,
            linestyle=linestyle,
            color=color,
            **kwargs
        )
        ax.add_patch(arc)

