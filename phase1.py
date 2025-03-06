from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
import os
import json
import cv2
import random
from PIL import Image, ImageFile

# Import some common detectron2 utilities
# this has pretrained models and other utilities
from detectron2 import model_zoo
# this has the running infernce function
from detectron2.engine import DefaultPredictor
# this has the function for configuring the model
from detectron2.config import get_cfg
# this has the function for visualizing , draawing the bounding boxes on the image
from detectron2.utils.visualizer import Visualizer
# this is for register and managing datasets
from detectron2.data import MetadataCatalog, DatasetCatalog
#Defines formats for bounding boxes.
from detectron2.structures import BoxMode
#For evaluating model performance.
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from multiprocessing import freeze_support

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Class mappings (adjust as necessary)
class_ids = {
    "person": 0,
    "man": 1,
    "woman": 2,
    "child": 3
}

def get_LVMHPV2_dicts(img_dir):
    """
    Function to load dataset annotations from a json file located in the img_dir
    and return a list of dictionaries in the format expected by Detectron2.
    """
    json_file = os.path.join(img_dir, "data_list.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    reduced_img_dir = os.path.join(img_dir, "")
    dataset_dicts = []
    corrupted_images = []

    for idx, v in enumerate(imgs_anns):
        print(f"Processing image {idx + 1}/{len(imgs_anns)}")
        record = {}
        try:
            filename = os.path.join(reduced_img_dir, v['filepath'].replace('../LV-MHP-v2/', ''))
            print(f"Loading image: {filename}")

            # Verify image using Pillow
            with Image.open(filename) as img_check:
                img_check.verify()

            # Read image dimensions with OpenCV
            img = cv2.imread(filename, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Image {filename} is empty or unreadable.")

            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            corrupted_images.append(filename)
            continue

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for box in v["bboxes"]:
            obj = {
                "bbox": [box["x1"], box["y1"], box["x2"], box["y2"]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_ids[box["class"]],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if corrupted_images:
        print(f"Found {len(corrupted_images)} corrupted images:")
        for img in corrupted_images:
            print(f"Corrupted image: {img}")
    else:
        print("No corrupted images found.")

    return dataset_dicts

if __name__ == '__main__':
    freeze_support()  # Necessary for Windows-based multiprocessing

    # Register train and validation datasets with the correct paths
    for d in ["train", "val"]:
        DatasetCatalog.register("LVMHPV2_" + d, lambda d=d: get_LVMHPV2_dicts(f"../LV-MHP-v2/{d}"))
        MetadataCatalog.get("LVMHPV2_" + d).set(thing_classes=["person", "man", "woman", "child"])

    LVMHP_meta = MetadataCatalog.get("LVMHPV2_train")

    # Visualize a few samples from the dataset (optional)
    print("Visualizing samples from the training set...")
    dataset_dicts = get_LVMHPV2_dicts("../LV-MHP-v2/train")
    print(f"Loaded {len(dataset_dicts)} images for training.")  # Debugging line

    if len(dataset_dicts) < 3:
        print("Not enough images for visualization.")
    else:
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=LVMHP_meta, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow('Image Window', out.get_image()[:, :, ::-1])
            cv2.waitKey(0)

    # Configuration for training the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
    cfg.DATASETS.TEST = ()  # no validation set for this example
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Initialize from pre-trained model
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust this based on GPU availability
    cfg.SOLVER.BASE_LR = 0.00025  # Good starting learning rate
    cfg.SOLVER.MAX_ITER = 200    # Total number of iterations
    cfg.SOLVER.STEPS = []        # No learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Smaller batch size (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Number of classes, including person, man, woman, child
    cfg.OUTPUT_DIR = "./phase1"  # Directory to save model checkpoints

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize and train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # Inference should use the config with parameters used during training
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # Perform inference on some validation data (optional)
    dataset_dicts = get_LVMHPV2_dicts("../LV-MHP-v2/val")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=LVMHP_meta, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("window", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    # Evaluation (optional)
    evaluator = COCOEvaluator("LVMHPV2_val", output_dir="./phase1")
    val_loader = build_detection_test_loader(cfg, "LVMHPV2_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))



