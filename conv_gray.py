import cv2
import os
from algorithm1 import calc_difference_mask, remove_false_alarms_one_image, morph_operations

def sliding_chunks(lst, chunk_size):
    # Loop to create sliding windows
    for i in range(len(lst)):
        # Extract a chunk from the list
        chunk = lst[i:i + chunk_size]
        # If the chunk is empty, break out of the loop
        if not chunk:
            break
        yield chunk

dataset_location = 'vocrs_dataset'
max_width = 0
max_height = 0

for set_name in ['test', 'train', 'val']:
    set_loc = os.path.join(dataset_location, set_name)
    gray_dir = os.path.join(set_loc, 'images_ir')
    os.makedirs(gray_dir, exist_ok=True)
    images_path = os.path.join(set_loc, 'images')
    images_list = sorted(os.listdir(images_path))
    images_path = [os.path.join(images_path, image) for image in images_list if image != '.DS_Store']
    frames = []
    framess = []
    # frames = [cv2.imread(image, cv2.IMREAD_COLOR) for image in images_path]
    w = 0
    h = 0
    for image in images_path:
        frame = cv2.imread(image, cv2.IMREAD_COLOR) 
        wid, hei, _ = frame.shape
        if w != wid or h != hei:
            if framess != []:
                frames.append(framess)
            framess = []
            w = wid
            h = hei
        framess.append(frame)
    
    m = len(frames)
    masks = []
    counter = 0
    for fr in frames:
        img3_list = [chunk for chunk in sliding_chunks(fr, 3)]
        
        for img_list in img3_list:
            diff, mask = calc_difference_mask(img_list)
            masks.append(mask)

            mask = morph_operations(mask)
            new_mask = remove_false_alarms_one_image(mask)
            cv2.imwrite(os.path.join(gray_dir, f"{images_list[counter]}"), new_mask)
            counter +=1
    # rgb_dir = os.path.join(set_loc, 'images_rgb')
    # os.rename(images_path, rgb_dir)