import os
import cv2
import xml.etree.ElementTree as ET

# Function to parse XML annotation file
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append((obj_name, (xmin, ymin, xmax, ymax)))
    return objects

# Function to annotate images and save as video
def annotate_images(images_folder, annotations_folder, output_video):
    image_files = sorted(os.listdir(images_folder))
    xml_files = sorted(os.listdir(annotations_folder))

    print(len(image_files))
    print(len(xml_files))

    # Get the XML files corresponding to images
    assert len(image_files) == len(xml_files), "Number of images and XML files must be equal"

    # Open the first image to get dimensions
    first_image = cv2.imread(os.path.join(images_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # adjust codec as per your need (e.g., 'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 10.0, (width, height))

    for img_file, xml_file in zip(image_files, xml_files):
        image_path = os.path.join(images_folder, img_file)
        xml_path = os.path.join(annotations_folder, xml_file)

        # Read image
        img = cv2.imread(image_path)

        # Parse XML annotation
        objects = parse_annotation(xml_path)

        # Draw bounding boxes on the image
        for obj_name, (xmin, ymin, xmax, ymax) in objects:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, obj_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the annotated frame to video
        out.write(img)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    images_folder = '/home/diana/mot/VISO_paper/coco/car/test2017'
    annotations_folder = '/home/diana/mot/VISO_paper/coco/car/Annotations/test2017'
    output_video = '/home/diana/mot/VISO_paper/fldout/annotated_cars_test.mp4'

    annotate_images(images_folder, annotations_folder, output_video)
    print(f'Annotated video saved as {output_video}')


