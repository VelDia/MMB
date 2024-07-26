import os
import numpy as np
import cv2
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd
import time
start_time = time.time()

def load_images_from_folder(images_folder):

    # List all image files in the folder
    image_files = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')])
    max_height = 0
    max_width = 0
    max_layers = 0
    for image_file in image_files:
        frame = cv2.imread(image_file)
        img_height, img_width, layers = frame.shape
        if img_height > max_height:
            max_height = img_height
        if img_width > max_width:
            max_width = img_width
        if layers > max_layers:
            max_layers = layers
    print(max_height, max_width, max_layers)
    end_time = time.time()

    print("Time elapsed in loading images: ", end_time-start_time) 
    frames = [cv2.resize(cv2.imread(image_file), (max_height, max_width)) for image_file in image_files]
    return frames

def construct_data_matrix(images):
    # Flatten images and construct the data matrix
    end_time = time.time()

    print("Time elapsed in construct data matrix: ", end_time-start_time) 
    flattened_images = [img.flatten() for img in images]
    V = np.column_stack(flattened_images)
    end_time = time.time()

    print("Time elapsed after construction data matrix: ", end_time-start_time) 
    return V

def fRMC(V, rank, max_iter=100, tol=1e-5):
    # Initial low-rank approximation using SVD
    U, S, VT = randomized_svd(V, n_components=rank, random_state=None)
    B = np.dot(U, np.dot(np.diag(S), VT))
    end_time = time.time()

    print("Time elapsed in fRMC before loop: ", end_time-start_time) 
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
    end_time = time.time()

    print("Time elapsed after loop in fRMC: ", end_time-start_time) 
    return B

def estimate_rank (V, energy_threshold=0.9):
    U, S, VT = randomized_svd(V, n_components=min(V.shape)-1, random_state=None)
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    rank = np.searchsorted(cumulative_energy, energy_threshold) + 1
    return rank

def main():
    folder = 'mot/VISO_paper/coco/train/test2017/'
    images = load_images_from_folder(folder)
    V = construct_data_matrix(images)
    
    # Assuming we know the rank of the low-rank matrix
    rank = estimate_rank(V)
    
    # Solve for the low-rank background matrix
    B = fRMC(V, rank)

    # Reshape the columns of B back into image format
    background_images = [B[:, i].reshape(images[0].shape) for i in range(B.shape[1])]
    end_time = time.time()

    print("Time elapsed: ", end_time-start_time) 
    # Save the background images
    output_folder = 'output_background_images'
    os.makedirs(output_folder, exist_ok=True)
    for i, bg_img in enumerate(background_images):
        cv2.imwrite(os.path.join(output_folder, f'bg_img_{i}.png'), bg_img)
    end_time = time.time()

    print("Time elapsed: ", end_time-start_time) 

if __name__ == "__main__":
    main()