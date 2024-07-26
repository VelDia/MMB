import cv2
import os
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd

# Calculate the number of observation matrices
def calc_observ_matrix(frames):
    M = len(frames) 
    L = 4
    f = 10

    N = M/(L * f)
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


def output_candidate_motion_pixels(foreground, output_folder, idx):
    output_path = os.path.join(output_folder, f'foreground_{idx}.png')
    cv2.imwrite(output_path, foreground.astype(np.uint8) * 255)
