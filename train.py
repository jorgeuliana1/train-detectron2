# Default libs
import json, os, sys, random, time, cv2
from datetime import datetime

# Detectron2 modules
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo

# Our modules
from dataset import Dataset

# BoxMode is an enum that tells how the bounding box coordinates are read.
# DatasetCatalog and MetadataCatalog are responsible for keeping the dataset register.
# get_cfg allow us to start a cfg.
# Visualizer contains some visualization tools (as the name suggests), it is not gonna be really used here.
# DefaultTrainer is the model trainer we are going to use.
# model_zoo is a really useful tool that allow us to quickly export cfg and ckpt files from the web.

# Getting the path to configuration files (from command-line):
args = sys.argv
dataset_info_file_path = args[1] # This file will allow us to configure the dataset
train_info_file_path = args[2] # This file will allow us to configure the train

# Defining a few constants:
dataset_name = "m" # If you change this name errors may occur.
# TODO: Find out why the only dataset name accepted is "m"

# Importing our dataset:
train_dataset = Dataset(dataset_info_file_path, "TRAIN")

# Defining our `get_dicts` function:
def get_dicts():
    return train_dataset.get()
# The `get_dicts` function is a must, without it we can't register our dataset to the catalog

# Inserting our dataset into the DatasetCatalog (necessary if we want to use it)
DatasetCatalog.register(dataset_name, get_dicts)
my_dataset = MetadataCatalog.get(dataset_name)
my_dataset.image_root = train_dataset.base_dir
my_dataset.thing_classes = list(train_dataset.categories_dict.keys())

"""
The following commented code is for testing purposes only,
it allows us to verify if everything is alright with our dataset register.
"""
"""
dataset_dicts = DatasetCatalog.get(dataset_name)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("myimage.jpg", vis.get_image()[:, :, ::-1])
"""

# Opening the train configuration file:
with open(train_info_file_path, "r") as f:
    train_settings = json.load(f)

# Defining the output directory:
default_output_directory = "./output" # Every output will be under './output'.
user_set_output_directory = train_settings["OUTPUT_DIR"]
train_output_directory = os.path.join(default_output_directory, user_set_output_directory)

# Setting the CFG file output path:
cfg_file_output = os.path.join(train_output_directory, "cfg.yaml")

# Setting up the CFG:
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.DATASETS.TRAIN = (dataset_name)
cfg.DATASETS.TEST = () # No test dataset in use right now
cfg.DATALOADER.NUM_WORKERS = train_settings["NUM_WORKERS"]
cfg.MODEL_WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = train_settings["IMS_PER_BATCH"]
cfg.SOLVER.BASE_LR = train_settings["BASE_LR"]
cfg.SOLVER.MAX_ITER = (train_settings["MAX_ITER"])
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (train_settings["BATCH_SIZE_PER_IMAGE"])
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(my_dataset.thing_classes)  # Number of classes is variable.
cfg.OUTPUT_DIR = train_output_directory # User set output directory (under ./output/)

# Setting up the output directory:
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # Creating the output directory
with open(cfg_file_output, "w") as f: f.write(str(cfg)) # Saving the CFG file (may be useful in the future)

# Training our neural newtwork:
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
