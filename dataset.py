"""
We use annotations in the format "IMAGE_PATH, X0, Y0, X1, Y1, CATEGORY",
Detectron2 uses Coco-formated datasets, this file contains everything that
allow us to outperform this difference.
"""

import json, os
from detectron2.structures import BoxMode

class Dataset:

    def _set_categories_id(self):

        # This function allows you to configure a dictionary in the coco format

        # Opening the 'classes.json' file that contains classes info:
        with open(self.classes_file, "r") as f:
            j = json.load(f)

        # Getting the classes and removing the '__background__' element
        # COCO may have a different approach on background, we want to avoid problems
        categories = list(j.keys())
        if "__background__" in categories:
            categories.remove("__background__")

        # Inserting the categories into our newly created dictionary:
        categories_dict = {}
        for i in range(len(categories)):
            categories_dict[categories[i]] = i

        # Setting the dataset variables
        self.categories_dict = categories_dict
        self.categories_num = len(self.categories_dict)
    
    def _set_coco_dicts(self):

        # This function sets a coco-format dictionary.

        categories_id = self.categories_dict

        # Getting the CSV file info:
        with open(self.annotations_file, "r") as f:
            lines = f.read().strip().split("\n")

        images_dict = {} # `images_dict` is a dictionary where the index is the image path
        
        # Reading the lines of the annotation (CSV) file:
        for line in lines:

            # Getting the CSV data:
            info = line.split(",")
            file_name = os.path.join(self.base_dir, info[0])
            x0, y0 = int(info[1]), int(info[2])
            x1, y1 = int(info[3]), int(info[4])
            category = info[5]
            width, height = self.dimensions

            # Converting the CSV-format data into COCO-format data
            annotation = {
                "bbox" : [x0, y0, x1, y1],
                "bbox_mode" : BoxMode.XYXY_ABS, # Need to change from string to enum when move to LCAD
                "category_id" : categories_id[category],
                "is_crowd" : 0,
            }
            if file_name in images_dict.keys():
                images_dict[file_name]["annotations"].append(annotation)
            else:
                images_dict[file_name] = {
                    "file_name" : file_name,
                    "height" : height,
                    "width" : width,
                    "image_id" : info[0], # Using the relative name of the image as ID
                    "annotations" : [annotation]
                }

        images_list = []
        for key in images_dict.keys():
            images_list.append(images_dict[key])

        self.images = images_list

    def __init__(self, settings_file_path, dataset_type):

        # Opening the dataset settings file:
        with open(settings_file_path, "r") as f:
            dataset_info = json.load(f)[dataset_type]

        # Getting the data
        self.base_dir = dataset_info["base_directory"]
        self.annotations_file = dataset_info["annotations_csv"]
        self.classes_file = dataset_info["classes_json"]
        self.dimensions = dataset_info["width"], dataset_info["height"]

        # Setting up the dataset
        self._set_categories_id()
        self._set_coco_dicts()

    def get(self):

        # Returns the already formated data

        return self.images

    def get_images_paths(self):

        # Returns a list containing the path to every image in the dataset.

        paths = [] # `paths` will contain the image paths
        
        # Looking into every image dict:
        for image in self.images:
            paths.append(image["file_name"]) # Getting the absolute path in every image

        return paths