import json, os, random, time, cv2 # Default libs

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

# Defining a few constants:
dataset_info_file_path = "dataset_info.json" # This file will allow us to configure the dataset
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

# Setting up the CFG:
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.DATASETS.TRAIN = (dataset_name)
cfg.DATASETS.TEST = () # No test dataset in use right now
cfg.DATALOADER.NUM_WORKERS = 0 # TODO: CHANGE THIS
cfg.MODEL_WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
cfg.SOLVER.BASE_LR = 0.02 # TODO: CHANGE THIS
cfg.SOLVER.MAX_ITER = (300)  # 300 iterations seems good enough, but you can certainly train longer. TODO: CHANGE THIS
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset. TODO: CHANGE THIS
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(my_dataset.thing_classes)  # Number of classes is variable. TODO: CHANGE THIS

# NOTE: Not every CFG setting is the most appropriate, we may fix this in a near future.

# Training our neural newtwork:
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # The output will be saved under "./output/"
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
