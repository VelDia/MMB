import os
import cv2
import math
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd

# Calculate the number of observation matrices
def calc_num_observ_matrix(frames):
    M = len(frames) 
    L = 4
    f = 10

    N = math.ceil(M/(L * f))
    return N

# Estimate the current background model based on LRMC (used test_LRMC_paper.py)
def construct_data_matrix(frames):
    flattened_images = [img.flatten() for img in frames]
    V = np.column_stack(flattened_images)
    return V

def estimate_rank (V, energy_threshold=0.9):
    U, S, VT = randomized_svd(V, n_components=min(V.shape)-1, random_state=None)
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    rank = np.searchsorted(cumulative_energy, energy_threshold) + 1
    return rank

def fRMC(V, rank, max_iter=100, tol=1e-5):

    # Initial low-rank approximation using SVD
    U, S, VT = randomized_svd(V, n_components=rank, random_state=None)
    B = np.dot(U, np.dot(np.diag(S), VT))
    for i in range(max_iter):

        # Compute the residual
        residual = V - B

        # Frobenius norm of the residual
        frob_norm_residual = np.linalg.norm(residual, 'fro')

        # Update B using the rank-1 approximation
        U, S, VT = svd(B + residual, full_matrices=False)
        B_new = np.dot(U[:, :rank], np.dot(np.diag(S[:rank]), VT[:rank, :]))

        # Check for convergence
        if np.linalg.norm(B_new - B, 'fro') / frob_norm_residual < tol:
            break

        B = B_new
        
    return B

def saving_roi_from_mask(mask, num_im):
    # min_area = 5
    # max_area = 80
    # min_aspect_ratio = 1.0
    # max_aspect_ratio = 6.0
    min_area = 4
    max_area = 324
    min_aspect_ratio = 0.25
    max_aspect_ratio = 6.0

    # List to store extracted ROIs
    rois = []
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x, y, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        aspect_ratio = width / height if height != 0 else 0  # Calculate aspect ratio

        # Check if component meets area and aspect ratio criteria
        if min_area <= area <= max_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:

            # Draw the component on the new mask
            new_mask[labels == label] = 255

            # Append ROI to list
            rois.append([int(num_im), int(label), int(x), int(y), int(height), int(width)]) #append coordinates as they appear in (ground truth) gt.txt
        # print(rois)
        
    
    # cv2.destroyAllWindows()
    return new_mask, rois