import numpy as np
import matplotlib.pyplot as plt

from ...core import Pose, Map, Sensor, Drawable, ControlInput

class Robot(Drawable):
  def __init__(self, pos: Pose, sensors: list[Sensor], m: Map):
    #sensors are the positions of the sensors relative to the robot's position
    self.pos = pos
    self.sensors = sensors
    for s in self.sensors:
        s.updatePosition(self.pos)

    self.map = m
    self.sensor_readings = [0] * len(sensors)

  def plotRobot2D(self, ax: plt.Axes, r: float | int = 1., linestyle: str = '-', linewidth: int | float = 2, label: str = "Robot", colors: list[str, str] = ['b', 'r']):
    """
    plots a circle at coordinates (x,y) and a line indicating the orientation
    theta in degrees. The radius r can be set to a custom value.
    """
    xx = self.pos.x + r * self.pos.orientation.cos
    yy = self.pos.y + r * self.pos.orientation.sin

    #draw orientation idicator (x,y) -> (xx,yy)
    ax.plot([self.pos.x,xx], [self.pos.y,yy], label=None, color=colors[0], linestyle=linestyle, linewidth=linewidth)

    #draw circle centered at robot position
    ax.add_patch(
        plt.Circle((self.pos.x, self.pos.y), r, color=colors[1], fill=False, linestyle=linestyle, label=label, linewidth=linewidth)
    )

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

  def updatePosition(self, new_pos: Pose):
    self.pos = new_pos

    for sensor in self.sensors:
      sensor.updatePosition(new_pos)

  def getSensorReadings(self, std_meas_noise: float = 0.5, **kwargs):
    self.sensor_readings = []

    for sensor in self.sensors:
      reading = sensor.getReading(self.map, std_meas_noise=std_meas_noise, **kwargs)

      self.sensor_readings.append(reading)

    return self.sensor_readings

  def draw(self, ax: plt.Axes, r: float | int = 0.5, linewidth: int | float = 1, label: str | None = None, linestyle='-', draw_sensor: bool = True, draw_sensor_readings:bool = False, colors: list[str, str] = ['b', 'r']):
    self.plotRobot2D(ax, r=r, linestyle=linestyle, label=label, linewidth=linewidth, colors=colors)

    # Draw the sensors

    if draw_sensor:
      for sensor in self.sensors:
        sensor.draw(ax, draw_sensor_readings=draw_sensor_readings)

  def getSensorPositions(self):
    sensor_positions = []
    for sensor in self.sensors:
      sensor_positions.append(sensor.abs_pos)
      
    return sensor_positions

  def applyControl(self, u: ControlInput, motion_noise: bool = True) -> ControlInput:
    new_pos, u_eff = u.applyControl(self.pos, motion_noise)
    
    self.updatePosition(new_pos)

    return u_eff
