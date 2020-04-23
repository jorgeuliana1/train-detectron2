# Default libs
import json, os, sys, random, time, cv2
from datetime import datetime

# Detectron2 modules
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor

# Our modules:
from dataset import Dataset

# Getting arguments from command-line:
args = sys.argv
dataset_info_path = args[1]
test_info_path = args[2]

# Getting test dataset data:
dataset_name = "t" # Arbitrary name of the dataset.
dataset = Dataset(dataset_info_path, "TEST")
images_paths = dataset.get_images_paths()
# `images_paths` is a list containing the path to every annotated image in the dataset.

# Defining our `get_dicts` function:
def get_dicts():
    return dataset.get()
# The `get_dicts` function is a must, without it we can't register our dataset to the catalog

# Inserting our dataset into the DatasetCatalog (necessary if we want to use it)
DatasetCatalog.register(dataset_name, get_dicts)
my_dataset = MetadataCatalog.get(dataset_name)
my_dataset.image_root = dataset.base_dir
my_dataset.thing_classes = list(dataset.categories_dict.keys())

# Getting test info:
with open(test_info_path, "r") as f:
    test_info = json.load(f)

# Setting up cfg:
cfg = get_cfg()
cfg.merge_from_file(test_info["CFG_PATH"])
cfg.DATASETS.TEST = (dataset_name)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = test_info["SCORE_THRESH_TEST"]
cfg.MODEL.WEIGHTS = test_info["WEIGHTS_PATH"]

# Setting up output folder:
base_output_directory_name = "./infer_output"
output_folder_path = os.path.join(base_output_directory_name, test_info["OUTPUT_FOLDER"])
os.makedirs(output_folder_path, exist_ok=True) # Creating the output folder, if it doesn't exist.

# Saving the CFG to the output folder:
cfg_output_path = os.path.join(output_folder_path, "cfg.yaml") # Defining the output file path.
with open(cfg_output_path, "w") as f: f.write(str(cfg)) # Saving the cfg into the file.

# Predicting:
predictor = DefaultPredictor(cfg)
for image_path in images_paths: # Predicting for every image.
    img = cv2.imread(image_path) # Loading the image with opencv.
    outputs = predictor(img) # Predicting image annotations.

    # Getting prediction image path
    image_file_name = image_path.split("/")[-1] # We are interested at the last element of the path.
    output_path = os.path.join(output_folder_path, "images", output_path)
    
    # Saving the prediction image:
    visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])