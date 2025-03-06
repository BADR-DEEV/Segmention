




#this works with evalution 

from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
import os
import json
import cv2
import random
import numpy as np
from PIL import Image, ImageFile

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from multiprocessing import freeze_support



def get_masks_for_image(image_id, mask_dir):
    """
    Find all masks corresponding to a given image ID in the mask directory.
    """
    mask_prefix = f"{image_id}_"
    mask_files = [f for f in os.listdir(mask_dir) if f.startswith(mask_prefix) and f.endswith(".png")]
    return [os.path.join(mask_dir, mask_file) for mask_file in mask_files]

def get_dataset_dicts_with_masks_and_labels(json_path, img_dir, mask_dir):
    """
    Load dataset annotations from a JSON file, including masks and gender labels.
    Returns a list of dictionaries formatted for Detectron2.
    """
    print(f"Loading dataset from {json_path} + {img_dir} + {mask_dir}")
    with open(json_path, "r") as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    corrupted_images = []

    for idx, img_info in enumerate(imgs_anns["images"]):
        print(f"Processing image {idx + 1}/{len(imgs_anns['images'])}")
        record = {}
        try:
            # Full path to the image
            filename = os.path.join(img_dir, img_info['file_name'])
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

        # Add image info to the record
        record["file_name"] = filename
        record["image_id"] = img_info["id"]
        record["height"] = height
        record["width"] = width

        # Add annotations
        objs = []
        for ann in imgs_anns["annotations"]:
            if ann["image_id"] == img_info["id"]:
                cat = ann["category_id"]
                bbox = ann["bbox"]
                obj = {
                    "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": cat,
                    "segmentation": ann.get("segmentation", []),
                }

                # Add segmentation masks only for women (category_id == 2)
                # if cat == 1:
                #     masks = get_masks_for_image(img_info['file_name'].split('.')[0], mask_dir)
                #     if masks:
                #         obj["segmentation"] = masks
                #     else:
                #         print(f"Warning: No masks found for image {img_info['file_name']} and category 'woman'")

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









def overlay_boxes_and_masks(dataset_dicts, metadata):
    """
    Visualizes images with overlaid bounding boxes, labels, and blurred masks for a given dataset.
    Only applies masks and blurring to women (category_id == 2).
    """
    print("Visualizing samples with boxes and masks...")
    for d in random.sample(dataset_dicts, min(len(dataset_dicts), 20)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)

        for ann in d["annotations"]:
            bbox = ann["bbox"]
            category_id = ann["category_id"]

            # Draw bounding box and label for all categories
            label = metadata.thing_classes[category_id]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(img, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Apply masking only for females (category_id == 2)
            if category_id == 1:  # Female
                polygons = ann.get("segmentation", [])
                for polygon in polygons:
                    # Create a mask from the polygon
                    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [polygon], 1)

                    # Ensure the mask applies only within the bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    mask_cropped = mask[y1:y2, x1:x2]

                    # Create a solid color mask (e.g., black) where the mask is active
                    mask_colored = cv2.merge((mask_cropped, mask_cropped, mask_cropped))
                    blurred_img = cv2.GaussianBlur(img[y1:y2, x1:x2], (51, 51), 0)
                    img[y1:y2, x1:x2] = np.where(mask_colored > 0, blurred_img, img[y1:y2, x1:x2])

        # Display the image with bounding boxes and masked areas for females
        cv2.imshow("Boxes and Masks", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    freeze_support()
    # Paths to the split JSON files and image/mask directories
    train_json_path = "../t_annotations/full_instances_Train_updated.json"
    val_json_path = "../t_annotations/full_instances_Val_updated.json"
    img_dir_train = "../LV-MHP-v2/train/images"
    mask_dir_train = "../LV-MHP-v2/train/masks"
    img_dir_val = "../LV-MHP-v2/val/images"
    mask_dir_val = "../LV-MHP-v2/val/masks"

    # Register train and validation datasets
    DatasetCatalog.register(
        "LVMHPV2_train",
        lambda: get_dataset_dicts_with_masks_and_labels(train_json_path, img_dir_train, mask_dir_train)
    )
    DatasetCatalog.register(
        "LVMHPV2_val",
        lambda: get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val, mask_dir_val)
    )
    MetadataCatalog.get("LVMHPV2_train").set(
        thing_classes=["man", "woman"],
    )
    MetadataCatalog.get("LVMHPV2_val").set(
        thing_classes=["man", "woman",],
     
    )

    LVMHP_meta = MetadataCatalog.get("LVMHPV2_train")

    # Visualize samples from the training dataset with boxes and masks
    print("Visualizing samples from the training set with boxes and masks...")
    train_dataset_dicts = get_dataset_dicts_with_masks_and_labels(train_json_path, img_dir_train, mask_dir_train)
    val_dataset_dicts = get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val, mask_dir_val)

    overlay_boxes_and_masks(train_dataset_dicts, LVMHP_meta)

# Training configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
    cfg.DATASETS.TEST = ("LVMHPV2_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 100
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    
    cfg.MASK_ON = True   
    cfg.INPUT.MASK_FORMAT = "polygon" 

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # # Train the model
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # # Inference on validation data
    # # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # predictor = DefaultPredictor(cfg)

    # overlay_boxes_and_masks(val_dataset_dicts, MetadataCatalog.get("LVMHPV2_val"))

 


