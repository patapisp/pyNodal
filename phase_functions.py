__author__ = 'Chronis'
import numpy as np
import matplotlib.pyplot as plt


def cart2pol(x,y):
    """
    Takes cartesian (2D) coordinates and transforms them into polar.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def make_FQPM(width, height, xc=None, yc=None):
    """
    Creates a FQPM phase mask with dimensions height, width with center at xc,yc
    :param height:
    :param width:
    :param xc:
    :param yc:
    :return:
    """
    if (xc is None) or (yc is None):
        xc = int(width/2)
        yc = int(height/2)
    FQPM = np.zeros((width, height))
    FQPM[xc:, yc:] = 1
    FQPM[:xc, :yc] = 1
    FQPM[:xc, yc:] = 0
    FQPM[xc:, :yc] = 0
    return FQPM






