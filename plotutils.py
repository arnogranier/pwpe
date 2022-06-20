from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.patches import ConnectionPatch

def connect_subplots(ax1, ax2, arrowstyle="-|>,head_width=1,head_length=1", **kwargs):
    ycenter0 = sum(ax1.get_ylim())/2
    ycenter1 = sum(ax2.get_ylim())/2
    xright0 = ax1.get_xlim()[1]
    xleft1 = ax2.get_xlim()[0]
    return ax2.add_artist(ConnectionPatch(xyA=(xright0,ycenter0), xyB=(xleft1,ycenter1),
                                          coordsA="data", coordsB="data", axesA=ax1, axesB=ax2,
                                          lw=3, arrowstyle=arrowstyle, fc='black', **kwargs))

def gaussian2dplot(ax, m, cloud, points=None, vs=None):
    ax.scatter(cloud[:,0], cloud[:,1], color='gray', s=10)
    confidence_ellipse(cloud[:,0], cloud[:,1], ax, edgecolor='red', lw=3)
    ax.scatter(m[0], m[1], color='red', s=200)
    if points is not None:
        ax.scatter(points[0,0], points[0,1], color='black', marker='^', s=300)
        ax.scatter(points[1,0], points[1,1], color='black', marker='s', s=200)
        ax.plot([m[0], points[0,0]], [m[1], points[0,1]], '--k', lw='3')
        ax.plot([m[0], points[1,0]], [m[1], points[1,1]], '--k', lw='3')
    if vs is not None:
        ax.arrow(m[0], m[1], vs[0,0], vs[1,0], lw='3', ls='-', edgecolor='blue', head_width=.3, head_length=.3)
        ax.arrow(m[0], m[1], vs[0,1], vs[1,1], lw='3', ls='-', edgecolor='blue', head_width=.3, head_length=.3)
    return ax

# Confidence ellipse helper function from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)