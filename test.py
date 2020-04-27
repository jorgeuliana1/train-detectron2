# Default libs
import json, yaml, os, sys, random, time, cv2

# The progress bar is used in the last loop (You can modify it if you want to remove the dependency)
from progressbar import ProgressBar

# Detectron2 modules
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import PascalVOCDetectionEvaluator # Our evaluator
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

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
images = dataset.get()
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

os.makedirs(os.path.join(output_folder_path, "images"), exist_ok=True) # Creating the images folder, if it doesn't exist.

# Predicting:
predictor = DefaultPredictor(cfg)
images_predictions = [] # This list will be used to create a JSON file at the end of the predictions.
pbar = ProgressBar()
for image in pbar(images): # Predicting for every image (Using a progress bar).
    image_path = image["file_name"]
    img = cv2.imread(image_path) # Loading the image with opencv.
    outputs = predictor(img) # Predicting image annotations.

    # Getting prediction image path
    image_file_name = image_path.split("/")[-1] # We are interested at the last element of the path.
    output_path = os.path.join(output_folder_path, "images", image_file_name)

    # Getting the prediction info:
    prediction_data = outputs["instances"] # That's an Instance object containing data about the prediction.
    prediction_bboxes = prediction_data.pred_boxes.tensor # Converting from Box object to torch.tensor
    prediction_classes = prediction_data.pred_classes # That's already a torch.tensor

    # Casting our data to default python structures:
    prediction_bboxes = prediction_bboxes.tolist()
    prediction_classes = prediction_classes.tolist()

    # Creating predicted data dictionaries:
    predictions_dicts = [
            { "category_id" : category, "bbox" : bbox }
            for category, bbox in zip(prediction_classes, prediction_bboxes)
            ]

    # Creating annotated data dictionaries:
    annotated_dicts = [ 
            { "category_id" : annotation["category_id"], "bbox" : annotation["bbox"] }
            for annotation in image["annotations"]
            ]

    # Creating an image dict:
    image_dict = {
            "image_name" : image_file_name,
            "predictions" : predictions_dicts,
            "annotations" : annotated_dicts
            }
    images_predictions.append(image_dict)
    
    # Saving the prediction image:
    # visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    # vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])

# Saving the dicts to a JSON file:
json_path = os.path.join(output_folder_path, "predictions.json")
with open(json_path, "w") as json_file:
    json.dump(images_predictions, json_file, indent=2)
