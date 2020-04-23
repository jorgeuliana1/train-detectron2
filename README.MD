# SCRIPTS FOR TRAINING AND TESTING USING DETECTRON2

## PURPOSE:

This repository is intended to contain scripts for training and testing neural networks using Detectron2 ([facebookresearch/detectron2](https://github.com/facebookresearch/detectron2/)).

I created it to use in my paper, but feel free to use it too.

## USAGE:

For training: `python3 train.py dataset_info.json train_info.json`

Where `dataset_info.json` may look like:
```json
{
    "TRAIN" : {
        "base_directory" : "foo/bar",
        "annotations_csv" : "foo/annotations.csv",
        "classes_json" : "foo/classes.json",
        "width" : 1280,
        "height" : 720
    },
    "TEST" : {
        "base_directory" : "foobar/bar",
        "annotations_csv" : "foobar/annotations.csv",
        "classes_json" : "foobar/classes.json",
        "width" : 1280,
        "height" : 720
    }
}
```

`"TEST"` and `"TRAIN"` tags refer to training and testing datasets respectively.
`annotations.csv` file is something like:
```csv
image_001.jpg,0,0,200,200,CATEGORY_01
image_001.jpg,200,200,500,500,CATEGORY_02
```
in a general way it is:
```csv
image_path(relative to base_directory),x0,y0,x1,y1,category
```
`x0, y0, x1, y1` follows the `BoxMode.XYXY_ABS` pattern.

`classes.json` file is something like:
```json
{
    "__background__" : 0,
    "CATEGORY_01" : 1,
    "CATEGORY_02" : 2,
    "CATEGORY_03" : 3
}
```
where the `"__background"` tag is optional.

The repository provides models for `train_info.json` and `dataset_info.json`.