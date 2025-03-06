# import torch
# from detectron2.utils.logger import setup_logger
# setup_logger()

# # Import necessary libraries
# import os
# import json
# import cv2
# import random
# import numpy as np
# from PIL import Image
# from multiprocessing import freeze_support

# # Import detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor, DefaultTrainer
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
# from detectron2.structures import BoxMode, PolygonMasks, BitMasks
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset


# import cv2
# import numpy as np
# import os
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog

# import os
# import json
# import cv2
# import numpy as np
# from multiprocessing import freeze_support
# from PIL import Image

# # Import Detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor, DefaultTrainer
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
# from detectron2.structures import BoxMode
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# import cv2
# import random
# import matplotlib.pyplot as plt
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog
# from detectron2.data import transforms as T
# from detectron2.data.dataset_mapper import DatasetMapper 
# from detectron2.data import build_detection_train_loader

# def overlay_masks_on_image(image_path, mask_paths, bbox=None, save_path=None):
#     """
#     Overlays masks onto an image and displays the result.

#     Args:
#         image_path (str): Path to the image file.
#         mask_paths (list): List of paths to mask images.
#         bbox (tuple, optional): Bounding box in (x1, y1, x2, y2) format.
#         save_path (str, optional): Path to save the overlaid image.
#     """
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not load image {image_path}")
#         return

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

#     # Create an overlay for the mask
#     overlay = np.zeros_like(image, dtype=np.uint8)

#     for mask_path in mask_paths:
#         # Load each mask
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"Warning: Could not load mask {mask_path}")
#             continue

#         # Apply color to the mask (random color)
#         mask_color = np.random.randint(100, 255, (1, 3), dtype=np.uint8).tolist()[0]
#         colored_mask = np.dstack([mask] * 3)  # Convert grayscale to 3-channel
#         colored_mask = (colored_mask / 255 * np.array(mask_color)).astype(np.uint8)

#         # Overlay the mask onto the image
#         overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

#     # Combine original image with overlay
#     masked_image = cv2.addWeighted(image, 1, overlay, 0.5, 0)

#     # Draw bounding box if provided
#     if bbox:
#         x1, y1, x2, y2 = map(int, bbox)
#         cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

#     # Show the final image
#     cv2.imshow("Image with Masks", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Save the image if save_path is provided
#     if save_path:
#         cv2.imwrite(save_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
#         print(f"Saved overlay image to {save_path}")


# import torch
# from detectron2.utils.logger import setup_logger
# setup_logger()





# # class AugmentedTrainer(DefaultTrainer):
# #     @classmethod
# #     def build_train_loader(cls, cfg):
# #         mapper = DatasetMapper(cfg, is_train=True, augmentations=[
# #             T.RandomBrightness(0.8, 1.2),
# #             T.RandomContrast(0.8, 1.2),
# #             T.RandomRotation(angle=[-30, 30]),
# #             T.RandomFlip(horizontal=True, vertical=False)
# #         ])
# #         return build_detection_train_loader(cfg, mapper=mapper)
# def get_dataset_dicts_with_masks_and_labels(json_path, img_dir):
#     """
#     Load dataset annotations from a JSON file, ensuring segmentation is properly formatted.
#     """
#     print(f"Loading dataset from {json_path} + {img_dir}")

#     with open(json_path, "r") as f:
#         imgs_anns = json.load(f)

#     dataset_dicts = []
#     corrupted_images = []

#     for idx, img_info in enumerate(imgs_anns["images"]):
#         print(f"Processing image {idx + 1}/{len(imgs_anns['images'])}")
#         record = {}

#         try:
#             filename = os.path.join(img_dir, img_info['file_name'])
#             img = cv2.imread(filename, cv2.IMREAD_COLOR)
#             if img is None:
#                 raise ValueError(f"Image {filename} is empty or unreadable.")

#             height, width = img.shape[:2]
#         except Exception as e:
#             print(f"Error processing image {filename}: {e}")
#             corrupted_images.append(filename)
#             continue

#         record["file_name"] = filename
#         record["image_id"] = img_info["id"]
#         record["height"] = height
#         record["width"] = width

#         objs = []
#         for ann in imgs_anns["annotations"]:
#             if ann["image_id"] == img_info["id"]:
#                 bbox = ann["bbox"]
#                 obj = {
#                     "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
#                     "bbox_mode": BoxMode.XYXY_ABS,
#                     "category_id": 0 if ann["category_id"] == 1 else 1,
#                     "gender_id": 0 if ann["category_id"] == 1 else 1
#                 }

#                 # Ensure segmentation is a nested list
#                 sem = ann.get("segmentation", [])
#                 if isinstance(sem, list) and sem and isinstance(sem[0], list):
#                     valid_polygons = [polygon for polygon in sem if len(polygon) >= 6]
#                     obj["segmentation"] = valid_polygons if valid_polygons else []
#                 else:
#                     obj["segmentation"] = []

#                 objs.append(obj)

#         record["annotations"] = objs
#         dataset_dicts.append(record)

#     return dataset_dicts


# def visualize_dataset_sample(dataset_dicts, metadata, sample_idx=None):
#     """
#     Visualize a random sample (or a specific sample) from the dataset.

#     Args:
#         dataset_dicts (list): The dataset in Detectron2 format.
#         metadata (MetadataCatalog): Metadata for categories.
#         sample_idx (int, optional): Index of a specific sample to visualize. Defaults to None (random).
#     """
#     # Pick a random sample if no index is provided
#     if sample_idx is None:
#         sample_idx = random.randint(0, len(dataset_dicts) - 1)

#     sample = dataset_dicts[sample_idx]
    
#     # Load image
#     img = cv2.imread(sample["file_name"])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Create visualizer
#     visualizer = Visualizer(img, metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
#     out = visualizer.draw_dataset_dict(sample)

#     # Display image with annotations
#     plt.figure(figsize=(10, 10))
#     plt.imshow(out.get_image())
#     plt.axis("off")
#     plt.show()

# def visualize_predictions(predictor, dataset_dicts, metadata, sample_idx=None):
#     """
#     Visualize model predictions on a dataset sample.
    
#     Args:
#         predictor: Trained Detectron2 predictor.
#         dataset_dicts (list): Dataset dictionary.
#         metadata (MetadataCatalog): Dataset metadata.
#         sample_idx (int, optional): Index of the sample to visualize. Defaults to None (random).
#     """
#     if sample_idx is None:
#         sample_idx = random.randint(0, len(dataset_dicts) - 1)

#     sample = dataset_dicts[sample_idx]
#     img = cv2.imread(sample["file_name"])
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Run inference
#     outputs = predictor(img)

#     # Visualize predictions
#     visualizer = Visualizer(img_rgb, metadata=metadata, scale=0.5)
#     out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

#     # Show the image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(out.get_image())
#     plt.axis("off")
#     plt.show()

# # Example Usage
# if __name__ == "__main__":
#     freeze_support()
    
#     train_json_path = "../t_annotations/full_instances_Train_updated.json"
#     img_dir_train = "../LV-MHP-v2/train/images"
#     val_json_path = "../t_annotations/full_instances_Val_updated.json"
#     img_dir_val = "../LV-MHP-v2/val/images"

#     # Register datasets
#     DatasetCatalog.register("LVMHPV2_train", lambda: get_dataset_dicts_with_masks_and_labels(train_json_path, img_dir_train))
#     DatasetCatalog.register("LVMHPV2_val", lambda: get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val))

#     MetadataCatalog.get("LVMHPV2_train").set(thing_classes=[ "man", "woman", ], gender_classes=["Male", "Female"])
#     MetadataCatalog.get("LVMHPV2_val").set(thing_classes=["man", "woman", ], gender_classes=["Male", "Female"])

#     # Load dataset
#     dataset_dicts = get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val)
#     metadata = MetadataCatalog.get("LVMHPV2_val")

#     # Visualize dataset sample
#     visualize_dataset_sample(dataset_dicts, metadata)

#     # Configure model
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
#     cfg.DATASETS.TEST = ("LVMHPV2_val",)
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Adjusted to match dataset (man, woman)
#     cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 2
#     cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.BASE_LR =0.0001
#     cfg.SOLVER.MAX_ITER = 100
#     cfg.INPUT.FORMAT = "BGR"
#     cfg.SOLVER.STEPS = [] 
#     augmentation = [
#     T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
#     T.RandomApply(T.RandomRotation(angle=[-10, 10]), prob=0.3),
#     T.RandomBrightness(0.1, 0.3),
#     T.RandomContrast(0.1, 0.3)
# ]
#     cfg.INPUT.AUGMENTATIONS = augmentation
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.SOLVER.STEPS = [] 
#     cfg.MASK_ON = True

#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     # Train the model
#     trainer = DefaultTrainer(cfg)
#     # AugmentedTrainer.build_train_loader(cfg)
#     trainer.resume_or_load(resume=False)
#     trainer.train()


#     # Load trained model for inference
#     # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     predictor = DefaultPredictor(cfg)

#     # Evaluate the model
#     evaluator = COCOEvaluator("LVMHPV2_val", cfg, output_dir="./output_phase4_newwwww")
#     val_loader = build_detection_test_loader(cfg, "LVMHPV2_val")
#     print("🚀 Running model evaluation...")
#     results = inference_on_dataset(predictor.model, val_loader, evaluator)
#     print("📊 Evaluation Results:", results)

#     # Visualize predictions
#     visualize_predictions(predictor, dataset_dicts, metadata)
    





# def overlay_boxes_and_masks(dataset_dicts, metadata):
#     """
#     Visualizes images with overlaid bounding boxes, labels, and blurred masks for a given dataset.
#     Only applies masks and blurring to women (category_id == 2).
#     """
#     print("Visualizing samples with boxes and masks...")
#     for d in random.sample(dataset_dicts, min(len(dataset_dicts), 20)):
#         img = cv2.imread(d["file_name"])
#         visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)

#         for ann in d["annotations"]:
#             bbox = ann["bbox"]
#             category_id = 0 if ann["category_id"] == 1 else 1

#             # Draw bounding box and label for all categories
#             label = metadata.thing_classes[category_id]
#             cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
#             cv2.putText(img, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Apply masking only for females (category_id == 2)
#             if category_id == 1:  # Female
#                 mask_paths = ann.get("segmentation", [])
#                 for mask_path in mask_paths:
#                     if os.path.exists(mask_path):
#                         # Read the mask
#                         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#                         if mask is not None:
#                             # Ensure the mask applies only within the bounding box
#                             x1, y1, x2, y2 = map(int, bbox)
#                             mask_cropped = mask[y1:y2, x1:x2]

#                             # Create a solid color mask (e.g., black) where the mask is active
#                             mask_colored = cv2.merge((mask_cropped, mask_cropped, mask_cropped))
#                             black_overlay = np.zeros_like(mask_colored, dtype=np.uint8)  # Black color
#                             img[y1:y2, x1:x2] = np.where(mask_colored > 0, black_overlay, img[y1:y2, x1:x2])
#                         else:
#                             print(f"Warning: Could not read mask {mask_path}")
#                     else:
#                         print(f"Warning: Mask path {mask_path} does not exist.")

#         # Display the image with bounding boxes and masked areas for females
#         cv2.imshow("Boxes and Masks", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# train_dataset_dicts = get_dataset_dicts_with_masks_and_labels(train_json_path, img_dir_train, mask_dir_train)
# overlay_boxes_and_masks(train_dataset_dicts, MetadataCatalog.get("LVMHPV2_train")) 

















import torch
from detectron2.utils.logger import setup_logger

setup_logger()

# Import necessary libraries
import os
import json
import cv2
import random
import numpy as np
from PIL import Image
from multiprocessing import freeze_support

# Import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode, PolygonMasks, BitMasks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


import cv2
import numpy as np
import os
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

import os
import json
import cv2
import numpy as np
from multiprocessing import freeze_support
from PIL import Image

# Import Detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import cv2
import random
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import build_detection_train_loader


def overlay_masks_on_image(image_path, mask_paths, bbox=None, save_path=None):
    """
    Overlays masks onto an image and displays the result.

    Args:
        image_path (str): Path to the image file.
        mask_paths (list): List of paths to mask images.
        bbox (tuple, optional): Bounding box in (x1, y1, x2, y2) format.
        save_path (str, optional): Path to save the overlaid image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

    # Create an overlay for the mask
    overlay = np.zeros_like(image, dtype=np.uint8)

    for mask_path in mask_paths:
        # Load each mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask {mask_path}")
            continue

        # Apply color to the mask (random color)
        mask_color = np.random.randint(100, 255, (1, 3), dtype=np.uint8).tolist()[0]
        colored_mask = np.dstack([mask] * 3)  # Convert grayscale to 3-channel
        colored_mask = (colored_mask / 255 * np.array(mask_color)).astype(np.uint8)

        # Overlay the mask onto the image
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

    # Combine original image with overlay
    masked_image = cv2.addWeighted(image, 1, overlay, 0.5, 0)

    # Draw bounding box if provided
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show the final image
    cv2.imshow("Image with Masks", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image if save_path is provided
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay image to {save_path}")


import torch
from detectron2.utils.logger import setup_logger

setup_logger()


class AugmentedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
                T.RandomRotation(angle=[-30, 30]),
                T.RandomFlip(horizontal=True, vertical=False),
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)


def get_dataset_dicts(json_path, img_dir):
    with open(json_path, "r") as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    for img_info in imgs_anns["images"]:
        record = {}
        filename = os.path.join(img_dir, img_info["file_name"])
        img = cv2.imread(filename)
        height, width = img.shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = img_info["id"]
        record["height"] = height
        record["width"] = width
        
        objs = []
        for ann in imgs_anns["annotations"]:
            if ann["image_id"] == img_info["id"]:
                bbox = ann["bbox"]
                category_id = ann["category_id"]
                obj = {
                    "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": category_id,
                    "segmentation": ann.get("segmentation", [])
                }
                objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts



def visualize_dataset_sample(dataset_dicts, metadata, sample_idx=None):
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset_dicts) - 1)
    sample = dataset_dicts[sample_idx]
    img = cv2.imread(sample["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    visualizer = Visualizer(img, metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = visualizer.draw_dataset_dict(sample)
    plt.figure(figsize=(10, 10))
    plt.imshow(out.get_image())
    plt.axis("off")
    plt.show()


def visualize_predictions(predictor, dataset_dicts, metadata, sample_idx=None):
    """
    Visualize model predictions on a dataset sample.

    Args:
        predictor: Trained Detectron2 predictor.
        dataset_dicts (list): Dataset dictionary.
        metadata (MetadataCatalog): Dataset metadata.
        sample_idx (int, optional): Index of the sample to visualize. Defaults to None (random).
    """
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset_dicts) - 1)

    sample = dataset_dicts[sample_idx]
    img = cv2.imread(sample["file_name"])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    outputs = predictor(img)

    # Visualize predictions
    visualizer = Visualizer(img_rgb, metadata=metadata, scale=0.5)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Show the image
    plt.figure(figsize=(10, 10))
    plt.imshow(out.get_image())
    plt.axis("off")
    plt.show()


# Example Usage
if __name__ == "__main__":
    print("jjj")
    freeze_support()

    train_json_path = "../t_annotations/full_instances_Train_updated.json"
    img_dir_train = "../LV-MHP-v2/train/images"
    val_json_path = "../t_annotations/full_instances_Val_updated.json"
    img_dir_val = "../LV-MHP-v2/val/images"

    # Register datasets
    DatasetCatalog.register(
        "LVMHPV2_train",
        lambda: get_dataset_dicts(train_json_path, img_dir_train),
    )
    DatasetCatalog.register(
        "LVMHPV2_val",
        lambda: get_dataset_dicts(val_json_path, img_dir_val),
    )

    MetadataCatalog.get("LVMHPV2_train").set(
        thing_classes=["man", "woman"],
        gender_classes=["Male", "Female"],
    )
    MetadataCatalog.get("LVMHPV2_val").set(
       thing_classes=["man", "woman" ],
        gender_classes=["Male", "Female"],
    )

    # Load dataset
    dataset_dicts = get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val)
    metadata = MetadataCatalog.get("LVMHPV2_val")

    # Visualize dataset sample
    visualize_dataset_sample(dataset_dicts, metadata)

    # Configure model
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
    cfg.DATASETS.TEST = ("LVMHPV2_val",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Adjusted to match dataset (man, woman)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 20
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.STEPS = []
    # cfg.INPUT.FORMAT = "BGR"
    augmentation = [
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    T.RandomApply(T.RandomRotation(angle=[-10, 10]), prob=0.3),
    T.RandomBrightness(0.1, 0.3),
    T.RandomContrast(0.1, 0.3)
]
    # cfg.INPUT.AUGMENTATIONS = augmentation
    # masks = true
    cfg.MODEL.MASK_ON = True


    
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # overlay_boxes_and_masks(dataset_dicts, metadata)


    # Train the model
    trainer = DefaultTrainer(cfg)
    # AugmentedTrainer.build_train_loader(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Load trained model for inference
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    # Evaluate the model
    evaluator = COCOEvaluator("LVMHPV2_val", cfg, output_dir="./output_phase4_n")
    val_loader = build_detection_test_loader(cfg, "LVMHPV2_val")
    print("🚀 Running model evaluation...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print("📊 Evaluation Results:", results)

    
def overlay_boxes_and_masks(dataset_dicts, metadata):
    """
    Visualizes images with overlaid bounding boxes, labels, and blurred masks for a given dataset.
    Only applies masks and blurring to women (category_id == 2).
    """
    print("Visualizing samples with boxes and masks...")
    for d in random.sample(dataset_dicts, min(len(dataset_dicts), 20)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE
        )

        for ann in d["annotations"]:
            bbox = ann["bbox"]
            category_id = ann["category_id"]

            # Draw bounding box and label for all categories
            label = metadata.thing_classes[category_id]
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                label,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Apply masking only for females (category_id == 2)
            if category_id == 1:  # Female
                mask_paths = ann.get("segmentation", [])
                for mask_path in mask_paths:
                    if os.path.exists(mask_path):
                        # Read the mask
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Ensure the mask applies only within the bounding box
                            x1, y1, x2, y2 = map(int, bbox)
                            mask_cropped = mask[y1:y2, x1:x2]

                            # Create a solid color mask (e.g., black) where the mask is active
                            mask_colored = cv2.merge(
                                (mask_cropped, mask_cropped, mask_cropped)
                            )
                            black_overlay = np.zeros_like(
                                mask_colored, dtype=np.uint8
                            )  # Black color
                            img[y1:y2, x1:x2] = np.where(
                                mask_colored > 0, black_overlay, img[y1:y2, x1:x2]
                            )
                        else:
                            print(f"Warning: Could not read mask {mask_path}")
                    else:
                        print(f"Warning: Mask path {mask_path} does not exist.")

        # Display the image with bounding boxes and masked areas for females
        cv2.imshow("Boxes and Masks", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# train_dataset_dicts = get_dataset_dicts_with_masks_and_labels(train_json_path, img_dir_train, mask_dir_train)
# overlay_boxes_and_masks(train_dataset_dicts, MetadataCatalog.get("LVMHPV2_train"))



    # Visualize predictions
    visualize_predictions(predictor, dataset_dicts, metadata)
    

