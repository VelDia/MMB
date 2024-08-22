import cv2
import os
def convert_folder_to_video(image_folder):
    video_name = os.path.join(image_folder, 'output_video.mp4')

    # Get a list of all JPEG files in the specified folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()
    # Read the first image to get dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video_path = os.path.join(image_folder, video_name)
    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

    # Write each image to the video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the VideoWriter
    video.release()

    # Clean up
    cv2.destroyAllWindows()

def convert_frames_to_video(images):

    # Make sure to run the program in the correct folder
    video_path = 'output_video.mp4'

    images.sort()
    height, width, layers = images[0].shape

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

    # Write each image to the video
    for image in images:
        video.write(image)

    # Release the VideoWriter
    video.release()

    # Clean up
    cv2.destroyAllWindows()

folder = '/Users/diana/Desktop/MMB/yolo_dataset/images'
convert_folder_to_video(folder)