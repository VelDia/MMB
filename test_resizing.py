import cv2
import os

def resize_image(image, target_height):
    (h, w) = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image

def main(input_folder, resized_folder, original_sizes_folder, target_height):
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    if not os.path.exists(original_sizes_folder):
        os.makedirs(original_sizes_folder)

    # Resize images and save to a new folder
    image_sizes = {}  # To store original sizes

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            if image is not None:
                original_height, original_width = image.shape[:2]
                image_sizes[filename] = (original_height, original_width)

                resized_image = resize_image(image, target_height)
                resized_path = os.path.join(resized_folder, filename)
                cv2.imwrite(resized_path, resized_image)
                # cv2.imshow('Resized', resize_image)

    # Resize images back to original size and save them
    for filename in image_sizes:
        original_height, original_width = image_sizes[filename]
        img_path = os.path.join(resized_folder, filename)
        image = cv2.imread(img_path)
        if image is not None:
            restored_image = cv2.resize(image, (original_width, original_height))
            restored_path = os.path.join(original_sizes_folder, filename)
            cv2.imwrite(restored_path, restored_image)
            # cv2.imshow('Restored', restored_image)

    print("Image processing completed!")

if __name__ == "__main__":
    input_folder = '/Users/diana/Desktop/MMB/mot/car/001/img'
    resized_folder = '/Users/diana/Desktop/MMB/mot/car/001/img/resized_folder'
    original_sizes_folder = '/Users/diana/Desktop/MMB/mot/car/001/img/original_sizes_folder'
    os.makedirs(resized_folder, exist_ok=True)
    os.makedirs(original_sizes_folder, exist_ok=True)
    target_height = 640  # Set the target height for resizing

    main(input_folder, resized_folder, original_sizes_folder, target_height)
