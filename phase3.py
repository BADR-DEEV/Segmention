from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
import os
import json
import random
from PIL import Image, ImageFile
import cv2
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

ImageFile.LOAD_TRUNCATED_IMAGES = True


EXCLUDED_PARTS = [0, 2]

def generate_instance_masks(parsing_mask, excluded_parts=EXCLUDED_PARTS):
  
    try:
        mask = np.isin(parsing_mask, excluded_parts, invert=True).astype(np.uint8)
        return mask * 255  
    except Exception as e:
        print(f"Error in generating masks: {e}")
        return None

def process_parsing_annotations(image_dir, parsing_dir, mask_dir):
    """
    Process parsing annotations for multiple files per image to generate instance masks.

    Args:
        image_dir (str): Path to the images directory.
        parsing_dir (str): Path to the parsing annotations directory.
        mask_dir (str): Path to save the generated masks.
    """
    os.makedirs(mask_dir, exist_ok=True)  # Create the masks directory if it doesn't exist.

    # Group annotation files by prefix (image filename without extension)
    parsing_files = [f for f in os.listdir(parsing_dir) if f.endswith(".png")]
    parsing_groups = {}
    for f in parsing_files:
        prefix = "_".join(f.split("_")[:-2])  # Extract prefix from the annotation file
        parsing_groups.setdefault(prefix, []).append(f)

    for prefix, files in parsing_groups.items():
        print(f"Processing prefix: {prefix}, total files: {len(files)}")
        for i, parsing_file in enumerate(files):
            parsing_path = os.path.join(parsing_dir, parsing_file)
            mask_save_path = os.path.join(mask_dir, f"{prefix}_{i+1}.png")  # Save masks with unique filenames

            # Read the parsing mask
            parsing_mask = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
            if parsing_mask is None:
                print(f"Error: Could not read parsing mask {parsing_path}")
                continue

            # Generate the instance mask excluding specific parts
            instance_mask = generate_instance_masks(parsing_mask)
            if instance_mask is None:
                print(f"Skipping mask generation for {parsing_path} due to errors.")
                continue

            # Save the generated mask
            cv2.imwrite(mask_save_path, instance_mask)
            print(f"Saved mask: {mask_save_path}")

def create_dataset_dict(image_dir, parsing_dir, mask_dir, data_list_path):
    """
    Create a Detectron2-compatible dataset dictionary.

    Args:
        image_dir (str): Path to the images directory.
        parsing_dir (str): Path to the parsing annotations directory.
        mask_dir (str): Path to the generated masks directory.
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: Detectron2-compatible dataset dictionary.
    """
    try:
        with open(data_list_path) as f:
            data_list = json.load(f)
            print(f"Loaded {len(data_list)} entries from {data_list_path}")
    except Exception as e:
        print(f"Error loading data list: {e}")
        return []

    dataset_dicts = []
    for idx, data in enumerate(data_list):
        print(f"Processing item {idx + 1}/{len(data_list)}")
        if "filepath" not in data:
            print(f"Warning: 'filepath' missing for item {idx}, skipping.")
            continue  # Skip this entry if 'filepath' is missing

        record = {
            "file_name": data["filepath"],
            "image_id": idx,
            "height": data.get("height", 0),
            "width": data.get("width", 0),
            "annotations": [],
        }

        for bbox in data.get("bboxes", []):
            print(f"Processing bbox: {bbox}")
            # Match masks using the prefix of the annotation file
            mask_prefix = "_".join(os.path.basename(bbox.get("ann_path", "")).split("_")[:-2])
            mask_paths = [
                os.path.join(mask_dir, f)
                for f in os.listdir(mask_dir)
                if f.startswith(mask_prefix)
            ]

            if not mask_paths:
                print(f"Warning: No masks found for prefix {mask_prefix}. Skipping bbox.")
                continue

            for mask_path in mask_paths:
                record["annotations"].append(
                    {
                        "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0,  # Adjust this for specific categories if needed
                        "segmentation": mask_path,  # Path to the generated mask
                    }
                )

        dataset_dicts.append(record)

        if idx % 100 == 0:  # Log progress every 100 items
            print(f"Processed {idx + 1}/{len(data_list)} entries")

    return dataset_dicts

# Paths
train_dir = "../LV-MHP-v2/train"
val_dir = "../LV-MHP-v2/val"
mask_dir_train = "../LV-MHP-v2/train/masks"
mask_dir_val = "../LV-MHP-v2/val/masks"

# Generate masks for train and val
# uncomment this for generating 
# print("Generating masks for train set...")
# process_parsing_annotations(
#     image_dir=os.path.join(train_dir, "images"),
#     parsing_dir=os.path.join(train_dir, "parsing_annos"),
#     mask_dir=mask_dir_train,
# )

# print("Generating masks for val set...")
# process_parsing_annotations(
#     image_dir=os.path.join(val_dir, "images"),
#     parsing_dir=os.path.join(val_dir, "parsing_annos"),
#     mask_dir=mask_dir_val,
# )

# Create dataset dictionaries
print("Creating dataset dictionaries...")
train_data_list_path = os.path.join(train_dir, "data_list.json")
val_data_list_path = os.path.join(val_dir, "data_list.json")

train_dataset = create_dataset_dict(
    image_dir=os.path.join(train_dir, "images"),
    parsing_dir=os.path.join(train_dir, "parsing_annos"),
    mask_dir=mask_dir_train,
    data_list_path=train_data_list_path,
)

val_dataset = create_dataset_dict(
    image_dir=os.path.join(val_dir, "images"),
    parsing_dir=os.path.join(val_dir, "parsing_annos"),
    mask_dir=mask_dir_val,
    data_list_path=val_data_list_path,
)

# Register datasets with Detectron2
DatasetCatalog.register("LVMHPV2_train", lambda: train_dataset)
MetadataCatalog.get("LVMHPV2_train").set(thing_classes=["person"])

DatasetCatalog.register("LVMHPV2_val", lambda: val_dataset)
MetadataCatalog.get("LVMHPV2_val").set(thing_classes=["person"])

print("Datasets registered successfully!")

# Visualize some samples
print("Visualizing samples...")
for d in random.sample(train_dataset, min(len(train_dataset), 3)):  # Avoid index error
    img = cv2.imread(d["file_name"])
    mask_path = d["annotations"][0].get("segmentation", None)
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            blended = cv2.addWeighted(img, 0.6, np.stack([mask] * 3, axis=-1), 0.4, 0)
            cv2.imshow("Blended Mask", blended)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Warning: Could not read mask {mask_path}")
    else:
        print(f"Warning: Mask path {mask_path} does not exist.")








# from detectron2.utils.logger import setup_logger
# setup_logger()

# # Import necessary libraries
# import os
# import json
# import random
# from PIL import Image, ImageFile
# import cv2
# import numpy as np
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.structures import BoxMode

# # Enable loading of truncated images
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define excluded parts (fcace, left hand, right hand)
# 3 is for sure for jakets 
# #1 arms and hair
# 2 for hands
# 4  chest or neck ?? ? 
# 5 unknown
#6 unknown
# 7 left foot or feets ? 
#8 again feet ? or shoes or legs ?
# 9 unknown 
#10 skirt or dress
#11?
# 12 13 14 15 16 17 sunglasses and tie
# EXCLUDED_PARTS = [0, 2]
# EXCLUDED_PARTS = [0,2]


# def generate_instance_masks(parsing_mask, excluded_parts=EXCLUDED_PARTS):
  
#     mask = np.isin(parsing_mask, excluded_parts, invert=True).astype(np.uint8)
#     return mask * 255 

# def process_parsing_annotations(image_dir, parsing_dir, mask_dir, max_images=50):
#     """
#     Process parsing annotations for multiple files per image to generate instance masks.

#     Args:
#         image_dir (str): Path to the images directory.
#         parsing_dir (str): Path to the parsing annotations directory.
#         mask_dir (str): Path to save the generated masks.
#         max_images (int): Maximum number of images to process.
#     """
#     os.makedirs(mask_dir, exist_ok=True)  # Create the masks directory if it doesn't exist.

#     # Group annotation files by prefix (image filename without extension)
#     parsing_files = [f for f in os.listdir(parsing_dir) if f.endswith(".png")]
#     parsing_groups = {}
#     for f in parsing_files:
#         prefix = "_".join(f.split("_")[:-2])  # Extract prefix from the annotation file
#         parsing_groups.setdefault(prefix, []).append(f)

#     processed_images = 0
#     for prefix, files in parsing_groups.items():
#         print(f"Processing prefix: {prefix}")
#         for i, parsing_file in enumerate(files):
#             if processed_images >= max_images:  # Stop processing after max_images
#                 print(f"Processed {max_images} images, stopping.")
#                 return

#             parsing_path = os.path.join(parsing_dir, parsing_file)
#             mask_save_path = os.path.join(mask_dir, f"{prefix}_{i+1}.png")  # Save masks with unique filenames

#             # Read the parsing mask
#             parsing_mask = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
#             if parsing_mask is None:
#                 print(f"Error: Could not read parsing mask {parsing_path}")
#                 break

#             # Generate the instance mask excluding specific parts
#             instance_mask = generate_instance_masks(parsing_mask)

#             # Save the generated mask
#             cv2.imwrite(mask_save_path, instance_mask)
#             print(f"Saved mask: {mask_save_path}")

#             processed_images += 1

# def create_dataset_dict(image_dir, parsing_dir, mask_dir, data_list_path, max_images=50):
#     """
#     Create a Detectron2-compatible dataset dictionary.

#     Args:
#         image_dir (str): Path to the images directory.
#         parsing_dir (str): Path to the parsing annotations directory.
#         mask_dir (str): Path to the generated masks directory.
#         data_list_path (str): Path to the JSON data list file.
#         max_images (int): Maximum number of images to process.

#     Returns:
#         list: Detectron2-compatible dataset dictionary.
#     """
#     with open(data_list_path) as f:
#         data_list = json.load(f)
#         print(f"Loaded data list from {data_list_path}")

#     dataset_dicts = []
#     processed_images = 0
#     for idx, data in enumerate(data_list):
#         if processed_images >= max_images:  # Stop after max_images
#             print(f"Processed {max_images} images, stopping.")
#             break

#         if "filepath" not in data:
#             print(f"Warning: 'filepath' missing for item {idx}, skipping.")
#             break  # Skip this entry if 'filepath' is missing

#         record = {
#             "file_name": data["filepath"],
#             "image_id": idx,
#             "height": data["height"],
#             "width": data["width"],
#             "annotations": [],
#         }

#         for bbox in data["bboxes"]:
#             # Match masks using the prefix of the annotation file
#             mask_prefix = "_".join(os.path.basename(bbox["ann_path"]).split("_")[:-2])
#             mask_paths = [
#                 os.path.join(mask_dir, f)
#                 for f in os.listdir(mask_dir)
#                 if f.startswith(mask_prefix)
#             ]

#             if not mask_paths:
#                 print(f"Warning: No masks found for prefix {mask_prefix}. Skipping.")
#                 break

#             for mask_path in mask_paths:
#                 record["annotations"].append(
#                     {
#                         "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
#                         "bbox_mode": BoxMode.XYXY_ABS,
#                         "category_id": 0,  # Adjust this for specific categories if needed
#                         "segmentation": mask_path,  # Path to the generated mask
#                     }
#                 )

#         dataset_dicts.append(record)
#         processed_images += 1

#     return dataset_dicts

# # Paths
# train_dir = "../LV-MHP-v2/train"
# val_dir = "../LV-MHP-v2/val"
# mask_dir_train = "../LV-MHP-v2/train/masks"
# mask_dir_val = "../LV-MHP-v2/val/masks"

# # Generate masks for train and val (limit to 10 images)
# print("Generating masks for train set...")
# process_parsing_annotations(
#     image_dir=os.path.join(train_dir, "images"),
#     parsing_dir=os.path.join(train_dir, "parsing_annos"),
#     mask_dir=mask_dir_train,
#     max_images=50  # Limit to 10 images
# )

# print("Generating masks for val set...")
# process_parsing_annotations(
#     image_dir=os.path.join(val_dir, "images"),
#     parsing_dir=os.path.join(val_dir, "parsing_annos"),
#     mask_dir=mask_dir_val,
#     max_images=50  # Limit to 10 images
# )

# # Create dataset dictionaries (limit to 10 images)
# print("Creating dataset dictionaries...")
# train_data_list_path = os.path.join(train_dir, "data_list.json")
# val_data_list_path = os.path.join(val_dir, "data_list.json")

# train_dataset = create_dataset_dict(
#     image_dir=os.path.join(train_dir, "images"),
#     parsing_dir=os.path.join(train_dir, "parsing_annos"),
#     mask_dir=mask_dir_train,
#     data_list_path=train_data_list_path,
#     max_images=50  # Limit to 10 images
# )

# val_dataset = create_dataset_dict(
#     image_dir=os.path.join(val_dir, "images"),
#     parsing_dir=os.path.join(val_dir, "parsing_annos"),
#     mask_dir=mask_dir_val,
#     data_list_path=val_data_list_path,
#     max_images=50  # Limit to 10 images
# )

# # Register datasets with Detectron2
# DatasetCatalog.register("LVMHPV2_train", lambda: train_dataset)
# MetadataCatalog.get("LVMHPV2_train").set(thing_classes=["person"])

# DatasetCatalog.register("LVMHPV2_val", lambda: val_dataset)
# MetadataCatalog.get("LVMHPV2_val").set(thing_classes=["person"])

# print("Datasets registered successfully!")

# # Visualize some samples
# print("Visualizing samples...")
# for d in random.sample(train_dataset, 10):
#     img = cv2.imread(d["file_name"])
#     if d["annotations"]:  # Check if there are any annotations
#         mask = cv2.imread(d["annotations"][0]["segmentation"], cv2.IMREAD_GRAYSCALE)
#         if mask is not None:
#             blended = cv2.addWeighted(img, 0.6, np.stack([mask] * 3, axis=-1), 0.4, 0)
#             cv2.imshow("Blended Mask", blended)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()










# from detectron2.utils.logger import setup_logger

# setup_logger()

# # Import necessary libraries
# import os
# import json
# import random
# from PIL import Image, ImageFile
# import cv2

# # Import detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.structures import BoxMode
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from multiprocessing import freeze_support
# import numpy as np

# # Specify excluded parts (example: hands and face)
# # 3 for fce , 7 and 8 for hands left right
# EXCLUDED_PARTS = [3, 7, 8]  # IDs for hands and face; adjust as per your dataset.


# # Function to generate instance segmentation masks
# def generate_instance_masks(parsing_mask, excluded_parts=EXCLUDED_PARTS):
#     """
#     Generate an instance segmentation mask excluding specific parts.

#     Args:
#         parsing_mask (numpy.ndarray): Original parsing mask (grayscale image).
#         excluded_parts (list): List of part IDs to exclude from the mask.

#     Returns:
#         numpy.ndarray: Instance segmentation mask.
#     """
#     mask = np.isin(parsing_mask, excluded_parts, invert=True).astype(np.uint8)
#     return mask * 255  # Scale to 0-255 for visualization or saving.


# # Function to process parsing annotations and save masks
# def process_parsing_annotations(image_dir, parsing_dir, mask_dir):
#     """
#     Process parsing annotations and generate instance segmentation masks.

#     Args:
#         image_dir (str): Path to the images directory.
#         parsing_dir (str): Path to the parsing annotations directory.
#         mask_dir (str): Path to save the generated masks.
#     """
#     os.makedirs(mask_dir, exist_ok=True)

#     parsing_files = [f for f in os.listdir(parsing_dir) if f.endswith(".png")]
#     for parsing_file in parsing_files:
#         parsing_path = os.path.join(parsing_dir, parsing_file)
#         mask_save_path = os.path.join(mask_dir, parsing_file)

#         # Read the parsing mask
#         parsing_mask = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
#         if parsing_mask is None:
#             print(f"Error: Could not read parsing mask {parsing_path}")
#             continue

#         # Generate the instance mask excluding specific parts
#         instance_mask = generate_instance_masks(parsing_mask)

#         # Save the generated mask
#         cv2.imwrite(mask_save_path, instance_mask)
#         print(f"Saved mask: {mask_save_path}")


# # Function to create Detectron2-compatible dataset
# def create_dataset_dict(image_dir, parsing_dir, mask_dir, data_list_path):
#     """
#     Create a Detectron2-compatible dataset dictionary.

#     Args:
#         image_dir (str): Path to the images directory.
#         parsing_dir (str): Path to the parsing annotations directory.
#         mask_dir (str): Path to the generated masks directory.
#         data_list_path (str): Path to the JSON data list file.

#     Returns:
#         list: Dataset dictionary.
#     """
#     with open(data_list_path) as f:
#         data_list = json.load(f)
#         print(f"Loaded data list from {data_list_path}")

#     dataset_dicts = []
#     for idx, data in enumerate(data_list):
#         if "filepath" not in data:
#             print(f"Warning: 'filepath' missing for item {idx}, skipping.")
#             continue  # Skip this entry if 'filepath' is missing
#         record = {
#             "file_name": data["filepath"],
#             "image_id": idx,
#             "height": data["height"],
#             "width": data["width"],
#             "annotations": [],
#             }

#         for bbox in data["bboxes"]:
#             mask_path = os.path.join(mask_dir, os.path.basename(bbox["ann_path"]))
#             if not os.path.exists(mask_path):
#                 print(f"Warning: Mask {mask_path} not found. Skipping.")
#                 continue

#             record["annotations"].append(
#                 {
#                     "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
#                     "bbox_mode": BoxMode.XYXY_ABS,
#                     "category_id": 0,  # Adjust for your specific category IDs.
#                     "segmentation": mask_path,
#                 }
#             )

#         dataset_dicts.append(record)

#     return dataset_dicts


# # Paths
# train_dir = "../LV-MHP-v2/train"
# val_dir = "../LV-MHP-v2/val"
# t_annotations_dir = "../t_annotations"
# mask_dir_train = "../LV-MHP-v2/train/masks"
# mask_dir_val = "../LV-MHP-v2/val/masks"

# # Generate masks for train and val
# print("Generating masks for train set...")
# process_parsing_annotations(
#     image_dir=os.path.join(train_dir, "images"),
#     parsing_dir=os.path.join(train_dir, "parsing_annos"),
#     mask_dir=mask_dir_train,
# )

# print("Generating masks for val set...")
# process_parsing_annotations(
#     image_dir=os.path.join(val_dir, "images"),
#     parsing_dir=os.path.join(val_dir, "parsing_annos"),
#     mask_dir=mask_dir_val,
# )

# # Create dataset dictionaries
# print("Creating dataset dictionaries...")
# train_data_list_path = os.path.join(train_dir, "data_list.json")
# val_data_list_path = os.path.join(val_dir, "data_list.json")

# train_dataset = create_dataset_dict(
#     image_dir=os.path.join(train_dir, "images"),
#     parsing_dir=os.path.join(train_dir, "parsing_annos"),
#     mask_dir=mask_dir_train,
#     data_list_path=train_data_list_path,
# )

# val_dataset = create_dataset_dict(
#     image_dir=os.path.join(val_dir, "images"),
#     parsing_dir=os.path.join(val_dir, "parsing_annos"),
#     mask_dir=mask_dir_val,
#     data_list_path=val_data_list_path,
# )

# # Register datasets with Detectron2
# DatasetCatalog.register("LVMHPV2_train", lambda: train_dataset)
# MetadataCatalog.get("LVMHPV2_train").set(thing_classes=["person"])

# DatasetCatalog.register("LVMHPV2_val", lambda: val_dataset)
# MetadataCatalog.get("LVMHPV2_val").set(thing_classes=["person"])

# print("Datasets registered successfully!")

# # Visualize some samples
# for d in random.sample(train_dataset, 3):
#     img = cv2.imread(d["file_name"])
#     mask = cv2.imread(d["annotations"][0]["segmentation"], cv2.IMREAD_GRAYSCALE)
#     if mask is not None:
#         blended = cv2.addWeighted(img, 0.6, np.stack([mask] * 3, axis=-1), 0.4, 0)
#         cv2.imshow("Blended Mask", blended)
#         cv2.waitKey(0)
