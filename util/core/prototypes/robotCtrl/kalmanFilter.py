import matplotlib.pyplot as plt

from util.core.prototypes import Drawable
from util.core.general import plotGaussian

class KalmanFilter(Drawable):
    def draw(self, ax: plt.Axes, draw_mu: bool = True, color: str | None = None, label: str | None = None, n_sigmas: int = 3, **kwargs):
        plotGaussian(ax, self.mu.as_array[:2, :], self.sigma_mat[:2, :2], n_sigmas, draw_mu=draw_mu, color=color, label=label, **kwargs)
