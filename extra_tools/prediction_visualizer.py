# This script will allow you to visualize the annotations on "predictions.json" file
# The input of this scripted is outputed by `test.py`

import cv2, numpy # Our script is highly dependant of opencv

import json, os, sys # Files, system and command-line arguments manipulation

def draw_bbox(image, x0, y0, x1, y1, label, color):

    # This function returns `image` with a bounding box limited by (x0, y0) and (x1, y1)

    # `image` : cv2 image
    # `x0`, `y0`, `x1`, `y1` : coordinates of the bounding box
    # `label` : the annotation category name
    # `color` : the bounding box margin color, (B, G, R) format

    # Line thickness:
    thickness = 3

    # Drawing the contour:
    out_image = cv2.rectangle(
        image,
        (int(x0), int(y0)),
        (int(x1), int(y1)),
        color, thickness
    )

    # Adding the class id to the image:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 255) # All red text
    out_image = cv2.putText(
        out_image, label,
        (int(x0), int(y0)),
        font, 1, text_color, 
        thickness, cv2.LINE_AA
    )

    return out_image

# Getting the arguments from the command-line:
args = {
    "predictions_json" : sys.argv[1],
    "base_directory" : sys.argv[2] # The directory where the images are stored.
}

# Loading the JSON file:
with open(args["predictions_json"], "r") as json_file:
    predictions = json.load(json_file)

# Visualization ouput directory
vis_out_dir = "./vis_output"
os.makedirs(vis_out_dir, exist_ok=True) # Creating the directory, if it doesn't exist

# Iterating through the images of JSON:
for image_info in predictions:

    # Getting the image "absolute" path:
    image_abs_path = os.path.join(args["base_directory"], image_info["image_name"])

    # Setting the visualization output path:
    image_out_path = os.path.join(vis_out_dir, image_info["image_name"])

    # Loading the image:
    image = cv2.imread(image_abs_path, cv2.IMREAD_UNCHANGED)

    # Drawing the predicted annotations at the image:
    predicted_margin_color = (255, 0, 0)
    for prediction in image_info["predictions"]:
        x0, y0, x1, y1 = prediction["bbox"]
        annotated_cat_id = prediction["category_id"]
        image = draw_bbox(
            image, x0, y0, x1, y1,
            str(annotated_cat_id),
            predicted_margin_color
        )

    # Drawing the original annotations at the image:
    original_margin_color = (0, 255, 0)
    for annotation in image_info["annotations"]:
        x0, y0, x1, y1 = annotation["bbox"]
        annotated_cat_id = annotation["category_id"]
        image = draw_bbox(
            image, x0, y0, x1, y1,
            str(annotated_cat_id),
            original_margin_color
        )

    # Saving the image at the directory:
    cv2.imwrite(image_out_path, image)


