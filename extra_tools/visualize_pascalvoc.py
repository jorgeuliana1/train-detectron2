import cv2, os, sys
import numpy as np

# This script allow us to visualize the annotations of a generated dataset.

def get_arguments():

    # Returns a dictionary containing arguments from the command-line

    return {
        "dataset_root" : sys.argv[1],
        "output_directory" : sys.argv[2]
    }

def get_annotations(images_list_path, base_xml_path):

    # Returns a list of dictionaries
    # Each dictionary of the list is an annotated image

    # Getting the list of files:
    images_xml_paths = []

    # Getting the "train.txt" images XML:
    with open(os.path.join(images_list_path, "train.txt"), "r") as images_list_file:
        lines = images_list_file.read().strip().split("\n")
        for line in lines:
            if line != "":
                images_xml_paths.append(line)

    # Getting the "test.txt" images XML:
    with open(os.path.join(images_list_path, "test.txt"), "r") as images_list_file:
        lines = images_list_file.read().strip().split("\n")
        for line in lines:
            if line != "":
                images_xml_paths.append(line)

    images_dict = {} # We use a dict to merge the annotations by image
    
    from lxml import etree, objectify


    # Converting the XML to CSV style:
    annotation_lines = []
    for xml_file_path in images_xml_paths:
        with open(base_xml_path.format(xml_file_path), "r") as xml_file:
            my_xml = etree.XML(xml_file.read())

        folder_index = 0
        filename_index = 1
        annotation_info_index = -1

        image_abs_path = os.path.join(my_xml[folder_index].text, my_xml[filename_index].text)

        annotations = my_xml[annotation_info_index]
        annotation = annotations
        # for annotation in annotations:
        category_index = 0
        bbox_index = -1

        category = annotation[category_index].text
        bbox = annotation[bbox_index]
        x0, y0, x1, y1 = bbox

        csv_line = f"{image_abs_path},{x0.text},{y0.text},{x1.text},{y1.text},{category}"
        annotation_lines.append(csv_line)
                
    # Inserting each annotation to the images_dict
    for line in annotation_lines:

        # Getting annotation info:
        annotation_info = line.split(",")
        image_rel_path = annotation_info[0] # Image relative path
        x0, y0 = annotation_info[1], annotation_info[2]
        x1, y1 = annotation_info[3], annotation_info[4]
        category = annotation_info[5]

        annotation_dict = {
            "x0" : int(x0),
            "y0" : int(y0),
            "x1" : int(x1),
            "y1" : int(y1),
            "category" : category
        }

        # Inserting the annotation at the images_dict:
        if image_rel_path in images_dict.keys():
            images_dict[image_rel_path]["annotations"].append(annotation_dict)
        else:
            images_dict[image_rel_path] = {
                "image_name" : image_rel_path,
                "annotations" : [annotation_dict]
            }

    # Converting from images_dict to images_list:
    images_list = []
    for key in images_dict.keys():
        images_list.append(images_dict[key])

    return images_list

def draw_bounding_box(image, x0, y0, x1, y1, color=(0, 0, 255), thickness=3):

    return cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

def main():

    # Getting arguments:
    args = get_arguments()

    # Opening the input file:
    input_dir = args["dataset_root"]
    images_list_path = os.path.join(input_dir, "ImageSets", "Main")
    base_xml_path = os.path.join(input_dir, "Annotations", "{}.xml")
    images = get_annotations(images_list_path, base_xml_path)

    # Creating the output directory (if it doesn't exist):
    output_dir = args["output_directory"]
    images_dir = os.path.join(output_dir)
    os.makedirs(os.path.join(images_dir, "images"), exist_ok=True)

    for i in images:

        # Reading the image
        input_image_path = i["image_name"]
        out_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

        # Drawing every bounding box of the image in the output image
        for annotation in i["annotations"]:

            x0, y0 = annotation["x0"], annotation["y0"]
            x1, y1 = annotation["x1"], annotation["y1"]

            out_image = draw_bounding_box(out_image, x0, y0, x1, y1)

        # Saving the output image
        output_path = os.path.join(images_dir, os.path.split(i["image_name"])[-1])
        print(output_path)
        cv2.imwrite(output_path, out_image)

    return

if __name__ == "__main__":
    main()
