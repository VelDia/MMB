import os
import xml.etree.ElementTree as ET
import csv
import numpy as np

# Function to parse XML annotation file and collect detailed statistics
def parse_annotation_and_collect_stats(xml_file, stats):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        # Count total annotations
        stats['total_annotations'] += 1

        obj_name = obj.find('name').text

        # Count annotations by class
        if obj_name in stats['class_count']:
            stats['class_count'][obj_name] += 1
        else:
            stats['class_count'][obj_name] = 1

        # Collect bounding box statistics
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        width = xmax - xmin
        height = ymax - ymin
        bbox_area = width * height

        # Store side lengths for further calculations
        stats['bbox_widths'].append(width)
        stats['bbox_heights'].append(height)
        stats['bbox_areas'].append(bbox_area)

        # Calculate aspect ratio
        aspect_ratio = width / height if height != 0 else 0
        stats['bbox_aspect_ratios'].append(aspect_ratio)

    return stats

# Function to collect detailed statistics about annotations in a dataset
def collect_detailed_annotation_statistics(annotations_folder):
    xml_files = sorted(os.listdir(annotations_folder))
    stats = {
        'total_annotations': 0,
        'class_count': {},
        'bbox_widths': [],
        'bbox_heights': [],
        'bbox_areas': [],
        'bbox_aspect_ratios': []
    }

    for xml_file in xml_files:
        xml_path = os.path.join(annotations_folder, xml_file)
        stats = parse_annotation_and_collect_stats(xml_path, stats)

    # Calculate additional statistics
    stats['total_images'] = len(xml_files)
    stats['avg_annotations_per_image'] = stats['total_annotations'] / stats['total_images'] if stats['total_images'] > 0 else 0
    stats['min_bbox_width'] = min(stats['bbox_widths']) if stats['bbox_widths'] else 0
    stats['max_bbox_width'] = max(stats['bbox_widths']) if stats['bbox_widths'] else 0
    stats['avg_bbox_width'] = np.mean(stats['bbox_widths']) if stats['bbox_widths'] else 0
    stats['min_bbox_height'] = min(stats['bbox_heights']) if stats['bbox_heights'] else 0
    stats['max_bbox_height'] = max(stats['bbox_heights']) if stats['bbox_heights'] else 0
    stats['avg_bbox_height'] = np.mean(stats['bbox_heights']) if stats['bbox_heights'] else 0
    stats['min_bbox_area'] = min(stats['bbox_areas']) if stats['bbox_areas'] else 0
    stats['max_bbox_area'] = max(stats['bbox_areas']) if stats['bbox_areas'] else 0
    stats['avg_bbox_area'] = np.mean(stats['bbox_areas']) if stats['bbox_areas'] else 0
    stats['min_bbox_aspect_ratio'] = min(stats['bbox_aspect_ratios']) if stats['bbox_aspect_ratios'] else 0
    stats['max_bbox_aspect_ratio'] = max(stats['bbox_aspect_ratios']) if stats['bbox_aspect_ratios'] else 0
    stats['avg_bbox_aspect_ratio'] = np.mean(stats['bbox_aspect_ratios']) if stats['bbox_aspect_ratios'] else 0

    return stats

# Function to save statistics to a CSV file
def save_statistics_to_csv(folder_name, stats, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['statistic', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'statistic': 'Folder', 'value': str(folder_name)})
        writer.writerow({'statistic': 'Total images', 'value': stats['total_images']})
        writer.writerow({'statistic': 'Total annotations', 'value': stats['total_annotations']})
        writer.writerow({'statistic': 'Average annotations per image', 'value': stats['avg_annotations_per_image']})
        writer.writerow({'statistic': 'Minimum bounding box width', 'value': stats['min_bbox_width']})
        writer.writerow({'statistic': 'Maximum bounding box width', 'value': stats['max_bbox_width']})
        writer.writerow({'statistic': 'Average bounding box width', 'value': stats['avg_bbox_width']})
        writer.writerow({'statistic': 'Minimum bounding box height', 'value': stats['min_bbox_height']})
        writer.writerow({'statistic': 'Maximum bounding box height', 'value': stats['max_bbox_height']})
        writer.writerow({'statistic': 'Average bounding box height', 'value': stats['avg_bbox_height']})
        writer.writerow({'statistic': 'Minimum bounding box area', 'value': stats['min_bbox_area']})
        writer.writerow({'statistic': 'Maximum bounding box area', 'value': stats['max_bbox_area']})
        writer.writerow({'statistic': 'Average bounding box area', 'value': stats['avg_bbox_area']})
        writer.writerow({'statistic': 'Minimum bounding box aspect ratio', 'value': stats['min_bbox_aspect_ratio']})
        writer.writerow({'statistic': 'Maximum bounding box aspect ratio', 'value': stats['max_bbox_aspect_ratio']})
        writer.writerow({'statistic': 'Average bounding box aspect ratio', 'value': stats['avg_bbox_aspect_ratio']})
        writer.writerow({'statistic': '', 'value': ''})  # Empty row for separation
        writer.writerow({'statistic': 'Class distribution:', 'value': ''})
        for class_name, count in stats['class_count'].items():
            writer.writerow({'statistic': f"- {class_name}", 'value': count})

    print(f"Statistics saved to {output_csv}")

# Example usage
if __name__ == '__main__':
    
    output_csv = 'annotation_statistics.csv'

    detailed_annotation_stats = collect_detailed_annotation_statistics(annotations_folder)

    save_statistics_to_csv(annotations_folder, detailed_annotation_stats, output_csv)
