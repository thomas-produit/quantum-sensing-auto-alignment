"""

Author: Tranter Tech
Date: 2024
"""
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from zernike import RZern


def find_circular_FOV(image, radii_range, sigma=3):
    edges = canny(image, sigma=sigma)
    hough_res = hough_circle(edges, radii_range)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, radii_range, total_num_peaks=1)
    return np.array([cy[0], cx[0]]), radii[0]


def create_circular_mask(h, w, center=None, radius=None):
    # Taken and adapted from https://stackoverflow.com/a/44874588/5006763
    # Here I inverted x,y in the sense that center should have the form center = (y_center, x_center)
    if center is None: # use the middle of the image
        center = (int(h/2), int(w/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h-center[0], w-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)

    mask = np.invert(dist_from_center <= radius) # Need the inversion since Masked array mask the True regions
    return mask


def compute_zernike_decomposition(image, zernike_object):
    parameters = zernike_object.fit_cart_grid(image)[0]
    return parameters


def cost_evaluation(zernike_decomposition):
    piston = zernike_decomposition[0]
    mode_index = []
    for i in range(5):
        mode_index.append(RZern.nm2noll(i*2, 0) - 1)
    defocus = np.sum([zernike_decomposition[x] for x in mode_index])
    all_others = np.delete(zernike_decomposition, [0] + mode_index)

    piston_weight = 1
    defocus_weight = 5
    all_others_weight = 1

    piston_abs = abs(piston)
    defocus = abs(defocus)
    all_others_integrated = np.sum(np.abs(all_others))

    return -(piston_weight*piston_abs + defocus_weight*defocus - all_others_weight*all_others_integrated)