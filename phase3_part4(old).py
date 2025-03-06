from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
import os
import json
import cv2
import random
from PIL import Image, ImageFile
import numpy as np

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




# Class mappings for gender classification
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
import os
import json
import cv2
import random
from PIL import Image, ImageFile
import numpy as np

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




# Class mappings for gender classification
gender_ids = {
    "Male": 1,
    "Female": 2
}

# Class mappings for object detection
class_ids = {
    "man": 1,
    "woman": 2,
}

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
                bbox = ann["bbox"]
                obj = {
                    "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": ann["category_id"],
                }

                # Add segmentation masks only for women (category_id == 2)
                if ann["category_id"] == 2:
                    masks = get_masks_for_image(img_info['file_name'].split('.')[0], mask_dir)
                    if masks:
                        obj["segmentation"] = masks
                    else:
                        print(f"Warning: No masks found for image {img_info['file_name']} and category 'woman'")

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
    Only applies masks and blurring if the prediction is more likely to be a woman.
    """
    print("Visualizing samples with boxes and masks...")
    for d in random.sample(dataset_dicts, min(len(dataset_dicts), 3)):
        img = cv2.imread(d["file_name"])
        original_img = img.copy()  # Keep a copy of the original image for non-masked areas
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)

        for ann in d["annotations"]:
            bbox = ann["bbox"]
            category_id = ann["category_id"]

            # Draw bounding box and label
            label = metadata.thing_classes[category_id]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(img, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 2)

            # Apply masking for females (category_id == 2)
            if category_id == 1:
                mask_paths = ann.get("segmentation", [])
                for mask_path in mask_paths:
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask in grayscale
                        if mask is not None:
                            # Ensure the mask and image have the same dimensions
                            if mask.shape[:2] != img.shape[:2]:
                                print(f"Warning: Mask size does not match image size for {mask_path}.")
                                continue
                            
                            # Apply the mask: White areas of the mask (255) will keep the original image
                            # Black areas of the mask (0) will make the area transparent
                            masked_area = cv2.bitwise_and(original_img, original_img, mask=mask)
                            inverted_mask = cv2.bitwise_not(mask)
                            background_area = cv2.bitwise_and(img, img, mask=inverted_mask)

                            # Combine the masked area and background
                            img = cv2.add(masked_area, background_area)
                        else:
                            print(f"Warning: Could not read mask {mask_path}")
                    else:
                        print(f"Warning: Mask path {mask_path} does not exist.")

        # Display the image with the masked areas for females
        cv2.imshow("Boxes and Masks", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





if __name__ == "__main__":
    freeze_support()
    # Paths to the split JSON files and image/mask directories
    train_json_path = "../t_annotations/instances_Train_updated.json"
    img_dir_train = "../LV-MHP-v2/train/images"
    mask_dir__train = "../LV-MHP-v2/train/masks"
    #val
    # val_json_path = "../t_annotations/instances_Val_updated.json"
    # img_dir_val = "../LV-MHP-v2/val/images"
    # mask_dir_val = "../LV-MHP-v2/val/masks"

    # Register train and validation datasets
    DatasetCatalog.register(
        "LVMHPV2_train",
        lambda: get_dataset_dicts_with_masks_and_labels(train_json_path, img_dir_train, mask_dir__train)
    )
    # DatasetCatalog.register(
    #     "LVMHPV2_val",
    #     lambda: get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val, mask_dir_val)
    # )
    MetadataCatalog.get("LVMHPV2_train").set(
        thing_classes=["person", "man", "woman", "child"],
        gender_classes=["Male", "Female"]
    )
    # MetadataCatalog.get("LVMHPV2_val").set(
    #     thing_classes=["person", "man", "woman", "child"],
    #     gender_classes=["Male", "Female"]
    # )

    LVMHP_meta = MetadataCatalog.get("LVMHPV2_train")
    

    # Visualize samples from the training dataset with boxes and masks
    print("Visualizing samples from the training set with boxes and masks...")
    train_dataset_dicts = get_dataset_dicts_with_masks_and_labels(train_json_path, img_dir_train, mask_dir__train)
    overlay_boxes_and_masks(train_dataset_dicts, LVMHP_meta)

    # Training configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
    # cfg.DATASETS.TEST = ("LVMHPV2_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 100
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

# Inference on validation data
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    # val_dataset_dicts = get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val, mask_dir_val)
    # overlay_boxes_and_masks(val_dataset_dicts, MetadataCatalog.get("LVMHPV2_val"))

    # Evaluate the model
    # evaluator = COCOEvaluator("LVMHPV2_val",cfg,True, output_dir="./output2", )
    # val_loader = build_detection_test_loader(cfg, "LVMHPV2_val")
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))



# import os
# import json
# import random
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.structures import BoxMode
# from detectron2.engine import DefaultTrainer, DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.evaluation import COCOEvaluator
# from detectron2.data import build_detection_test_loader
# import cv2
# from PIL import Image
# from detectron2.utils.logger import setup_logger
# setup_logger()

# # Import necessary libraries
# import os
# import json
# import cv2
# import random
# from PIL import Image, ImageFile

# # Import some common detectron2 utilities
# # this has pretrained models and other utilities
# from detectron2 import model_zoo
# # this has the running infernce function
# from detectron2.engine import DefaultPredictor
# # this has the function for configuring the model
# from detectron2.config import get_cfg
# # this has the function for visualizing , draawing the bounding boxes on the image
# from detectron2.utils.visualizer import Visualizer
# # this is for register and managing datasets
# from detectron2.data import MetadataCatalog, DatasetCatalog
# #Defines formats for bounding boxes.
# from detectron2.structures import BoxMode
# #For evaluating model performance.
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# from detectron2.data import build_detection_test_loader
# from detectron2.engine import DefaultTrainer
# from detectron2.utils.visualizer import ColorMode
# from multiprocessing import freeze_support

# # Enable loading of truncated images
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # Enable loading of truncated images
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # Class mappings for gender classification
# gender_ids = {
#     "Male": 1,
#     "Female": 2
# }

# # Class mappings for object detection
# class_ids = {
#     "person": 0,
#     "man": 1,
#     "woman": 2,
#     "child": 3
# }

# def get_dataset_dicts(json_path, img_dir):
#     """
#     Load dataset annotations from a JSON file and return a list of dictionaries
#     in the format expected by Detectron2.
#     """
#     with open(json_path, "r") as f:
#         imgs_anns = json.load(f)

#     dataset_dicts = []
#     corrupted_images = []

#     for idx, img_info in enumerate(imgs_anns["images"]):
#         print(f"Processing image {idx + 1}/{len(imgs_anns['images'])}")
#         record = {}
#         try:
#             # Full path to the image
#             filename = os.path.join(img_dir, img_info['file_name'])
#             print(f"Loading image: {filename}")

#             # Verify image using Pillow
#             with Image.open(filename) as img_check:
#                 img_check.verify()

#             # Read image dimensions with OpenCV
#             img = cv2.imread(filename, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#             if img is None:
#                 raise ValueError(f"Image {filename} is empty or unreadable.")

#             height, width = img.shape[:2]
#         except Exception as e:
#             print(f"Error processing image {filename}: {e}")
#             corrupted_images.append(filename)
#             continue

#         # Add image info to the record
#         record["file_name"] = filename
#         record["image_id"] = img_info["id"]
#         record["height"] = height
#         record["width"] = width

#         # Add annotations
#         objs = []
#         for ann in imgs_anns["annotations"]:
#             if ann["image_id"] == img_info["id"]:
#                 bbox = ann["bbox"]
#                 obj = {
#                     "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
#                     "bbox_mode": BoxMode.XYXY_ABS,
#                     "category_id": ann["category_id"],  # Object class ID
#                     # "gender_label": gender_ids[ann["attributes"]["gender"]]  # Gender label
#                 }
#                 objs.append(obj)

#         record["annotations"] = objs
#         dataset_dicts.append(record)

#     if corrupted_images:
#         print(f"Found {len(corrupted_images)} corrupted images:")
#         for img in corrupted_images:
#             print(f"Corrupted image: {img}")
#     else:
#         print("No corrupted images found.")

#     return dataset_dicts

# if __name__ == "__main__":
#     # Paths to the split JSON files and image directory
#     # train_json_path = "t_annotations/instances_Train_Split.json"
#     train_json_path = "../t_annotations/instances_Train_Split.json"
#     # val_json_path = "t_annotations/instances_Val_Split.json"
#     val_json_path = "../t_annotations/instances_Val_Split.json"
#     img_dir = "../LV-MHP-v2/train/images"

#     # Register train and validation datasets
#     DatasetCatalog.register(
#         "LVMHPV2_train",
#         lambda: get_dataset_dicts(train_json_path, img_dir)
#     )
#     DatasetCatalog.register(
#         "LVMHPV2_val",
#         lambda: get_dataset_dicts(val_json_path, img_dir)
#     )
#     MetadataCatalog.get("LVMHPV2_train").set(
#         thing_classes=["person", "man", "woman", "child"],
#         gender_classes=["Male", "Female"]
#     )
#     MetadataCatalog.get("LVMHPV2_val").set(
#         thing_classes=["person", "man", "woman", "child"],
#         gender_classes=["Male", "Female"]
#     )

#     LVMHP_meta = MetadataCatalog.get("LVMHPV2_train")

# # Visualize a few samples from the training dataset
#     print("Visualizing samples from the training set...")
#     dataset_dicts = get_dataset_dicts(train_json_path, img_dir)
#     print(f"Loaded {len(dataset_dicts)} images for training.")  # Debugging line

#     if len(dataset_dicts) < 3:
#         print("Not enough images for visualization.")
#     else:
#         for d in random.sample(dataset_dicts, 3):
#             img = cv2.imread(d["file_name"])
#             visualizer = Visualizer(img[:, :, ::-1], metadata=LVMHP_meta, scale=0.5)
#             out = visualizer.draw_dataset_dict(d)
#             cv2.imshow('Image Window', out.get_image()[:, :, ::-1])
#             cv2.waitKey(0)

#     # Configuration for training the model
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
#     cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
#     cfg.DATASETS.TEST = ("LVMHPV2_val",)  # Add validation set
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Initialize from pre-trained model
#     cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust this based on GPU availability
#     cfg.SOLVER.BASE_LR = 0.00025  # Good starting learning rate
#     cfg.SOLVER.MAX_ITER = 200    # Total number of iterations
#     cfg.SOLVER.STEPS = []        # No learning rate decay
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Smaller batch size (default: 512)
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Number of classes for object detection

#     # Add gender classification to the model config
#     cfg.MODEL.GENDER_CLASSIFIER = {
#         "NUM_CLASSES": 2,  # Male and Female
#     }

#     # Create output directory if it doesn't exist
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     # Initialize and train the model
#     trainer = DefaultTrainer(cfg)
#     trainer.resume_or_load(resume=True)
#     trainer.train()

#     # Inference
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     predictor = DefaultPredictor(cfg)

#     # Perform inference on validation data
#     dataset_dicts = get_dataset_dicts(val_json_path, img_dir)
#     for d in random.sample(dataset_dicts, 3):
#         im = cv2.imread(d["file_name"])
#         outputs = predictor(im)
#         v = Visualizer(
#             im[:, :, ::-1],
#             metadata=MetadataCatalog.get("LVMHPV2_val"),
#             scale=0.5,
#             instance_mode=ColorMode.IMAGE_BW
#         )
#         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#         cv2.imshow("window", out.get_image()[:, :, ::-1])
#         cv2.waitKey(0)

#     # Evaluation
#     evaluator = COCOEvaluator("LVMHPV2_val", output_dir="./output2")
#     val_loader = build_detection_test_loader(cfg, "LVMHPV2_val")
#     print(inference_on_dataset(predictor.model, val_loader, evaluator))


# import json
# import random

# def split_sample_dataset(json_path, train_path, val_path, train_ratio=0.8):
#     """
#     Split the limited sample dataset into train and validation sets.

#     Args:
#         json_path (str): Path to the JSON file containing the sample dataset.
#         train_path (str): Path to save the train JSON file.
#         val_path (str): Path to save the validation JSON file.
#         train_ratio (float): Proportion of data to use for training (default: 0.8).
#     """
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     images = data["images"]
#     annotations = data["annotations"]

#     # Shuffle the images to ensure random splitting
#     random.shuffle(images)

#     # Calculate split index
#     split_idx = int(len(images) * train_ratio)

#     # Split images
#     train_images = images[:split_idx]
#     val_images = images[split_idx:]

#     # Create lookup for image IDs in each split
#     train_image_ids = {img["id"] for img in train_images}
#     val_image_ids = {img["id"] for img in val_images}

#     # Split annotations based on image IDs
#     train_annotations = [ann for ann in annotations if ann["image_id"] in train_image_ids]
#     val_annotations = [ann for ann in annotations if ann["image_id"] in val_image_ids]

#     # Save train and validation JSON files
#     train_data = {"images": train_images, "annotations": train_annotations}
#     val_data = {"images": val_images, "annotations": val_annotations}

#     with open(train_path, "w") as f:
#         json.dump(train_data, f, indent=4)

#     with open(val_path, "w") as f:
#         json.dump(val_data, f, indent=4)

#     print(f"Dataset split completed. Train: {len(train_images)} images, Val: {len(val_images)} images.")

# # Example usage
# split_sample_dataset(
#     json_path="../t_annotations/instances_Train.json",  # Path to the sample JSON file
#     train_path="../t_annotations/instances_Train_Split.json",  # Output path for train JSON
#     val_path="../t_annotations/instances_Val_Split.json",  # Output path for validation JSON
#     train_ratio=0.8  # Use 80% for training
# )