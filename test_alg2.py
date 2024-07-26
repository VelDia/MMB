import os
import numpy as np
import cv2
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd
from skimage.morphology import binary_dilation, binary_erosion, disk

def load_images_from_folder(folder, resize_dim=None):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if resize_dim is not None:
                img = cv2.resize(img, resize_dim)
            images.append(img)
    return images

def construct_data_matrix(images):
    flattened_images = [img.flatten() for img in images]
    V = np.column_stack(flattened_images)
    return V

def estimate_rank (V, energy_threshold=0.9):
    U, S, VT = randomized_svd(V, n_components=min(V.shape)-1, random_state=None)
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    rank = np.searchsorted(cumulative_energy, energy_threshold) + 1
    return rank

def fRMC(V, rank, max_iter=100, tol=1e-5):
    U, S, VT = randomized_svd(V, n_components=rank, random_state=None)
    B = np.dot(U, np.dot(np.diag(S), VT))

    for i in range(max_iter):
        residual = V - B
        frob_norm_residual = np.linalg.norm(residual, 'fro')

        U, S, VT = svd(B + residual, full_matrices=False)
        B_new = np.dot(U[:, :rank], np.dot(np.diag(S[:rank]), VT[:rank, :]))

        if np.linalg.norm(B_new - B, 'fro') / frob_norm_residual < tol:
            break

        B = B_new

    return B

def convert_to_binary(image, threshold=127):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def perform_morphological_operations(binary_image, operation='dilation', radius=2):
    selem = disk(radius)
    if operation == 'dilation':
        morphed_image = binary_dilation(binary_image, selem)
    elif operation == 'erosion':
        morphed_image = binary_erosion(binary_image, selem)
    return morphed_image

def output_candidate_motion_pixels(foreground, output_folder, idx):
    output_path = os.path.join(output_folder, f'foreground_{idx}.png')
    cv2.imwrite(output_path, foreground.astype(np.uint8) * 255)

def process_batch(folder, batch_size, resize_dim, output_folder):
    images = load_images_from_folder(folder, resize_dim)
    num_batches = len(images) // batch_size + (1 if len(images) % batch_size != 0 else 0)

    for i in range(num_batches):
        batch_images = images[i * batch_size:(i + 1) * batch_size]
        V = construct_data_matrix(batch_images)
        rank = estimate_rank(V)
        B = fRMC(V, rank)

        for j in range(B.shape[1]):
            background = B[:, j].reshape(batch_images[0].shape)
            foreground = batch_images[j] - background

            # Convert to binary image
            binary_image = convert_to_binary(foreground)

            # Perform morphological operations
            morphed_image = perform_morphological_operations(binary_image)

            # Output candidate motion pixels
            output_candidate_motion_pixels(morphed_image, output_folder, i * batch_size + j)

def main():
    images_folder = 'path_to_your_images_folder'
    output_folder = 'output_foreground_images'
    resize_dim = (128, 128)  # Resize images to 128x128
    batch_size = 10  # Process 10 images at a time

    os.makedirs(output_folder, exist_ok=True)
    process_batch(images_folder, batch_size, resize_dim, output_folder)

if __name__ == "__main__":
    main()