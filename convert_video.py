import cv2
import os
import time
# from itertools import chain
from mot.VISO_paper.convert_annot_video import annotate_images

start_time = time.time()

paths_dict = {
    'cars_test' : 'mot/VISO_paper/coco/car/test2017/',
    'planes_test' : 'mot/VISO_paper/coco/plane/test2017',
    'ships_test' : 'mot/VISO_paper/coco/ship/test2017',
    'trains_test' : 'mot/VISO_paper/coco/train/test2017',

    'cars_train' : 'mot/VISO_paper/coco/car/train2017/',
    'planes_train' : 'mot/VISO_paper/coco/plane/train2017',
    'ships_train' : 'mot/VISO_paper/coco/ship/train2017',
    'trains_train' : 'mot/VISO_paper/coco/train/train2017',

    'cars_val' : 'mot/VISO_paper/coco/car/val2017/',
    'planes_val' : 'mot/VISO_paper/coco/plane/val2017',
}

opath_vid = '/home/diana/mot/VISO_paper/fldout/videos'
if not os.path.exists(opath_vid):
    os.makedirs(opath_vid)

for video_name, image_folder in paths_dict.items():
    
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files.sort()

    height, width = None, None
    i = 0
    for image_file in image_files:
    # Read the image
        frame = cv2.imread(os.path.join(image_folder, image_file))
        # Get dimensions of the image
        img_height, img_width, layers = frame.shape

        # If dimensions are not initialized or dimensions have changed, update them
        if height is None or width is None or (img_height != height or img_width != width):
            height, width = img_height, img_width
            # Reinitialize VideoWriter with updated dimensions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            i +=1
            video = cv2.VideoWriter(os.path.join(opath_vid, video_name + str(i) + '.mp4'), fourcc, 10, (width, height))
            print(f"VideoWriter initialized with dimensions: {width}x{height}")

        # Write the frame to the video
        video.write(frame)

    # Release the VideoWriter
    video.release()
    print("Video writing completed: " + video_name)

    # Clean up
    cv2.destroyAllWindows()

end_time = time.time()

print("Time elapsed: ", end_time-start_time) 
