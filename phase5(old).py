import torch
import os
import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from detectron2 import model_zoo

# Path to test images folder
test_img_dir = "../LV-MHP-v2/test/images/"

def load_test_images(img_dir):
    dataset_dicts = []
    for idx, img_file in enumerate(os.listdir(img_dir)):
        record = {}
        record["file_name"] = os.path.join(img_dir, img_file)
        record["image_id"] = idx
        dataset_dicts.append(record)
    return dataset_dicts

test_dataset = load_test_images(test_img_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.OUTPUT_DIR = "./ph"

predictor = DefaultPredictor(cfg)

def evaluate_test_images(predictor, dataset):
    pred_labels, true_labels = [], []
    for data in dataset:
        img = cv2.imread(data["file_name"])
        if img is None:
            print(f"Failed to load image: {data['file_name']}")
            continue  # Skip this image

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        pred_labels.extend(instances.pred_classes.tolist())
    
    true_labels = [0] * len(pred_labels)  # Replace with actual labels if available

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    print(f"Classification Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, output_dir=cfg.OUTPUT_DIR)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    for data in dataset[:5]:
        img = cv2.imread(data["file_name"])
        if img is None:
            print(f"Failed to load image: {data['file_name']}")
            continue  # Skip this image

        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=0.5, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(10, 10))
        plt.imshow(v.get_image())
        plt.axis("off")
        plt.show()

evaluate_test_images(predictor, test_dataset)






# import torch
# from detectron2.utils.logger import setup_logger
# setup_logger()

# import os
# import json
# import cv2
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from multiprocessing import freeze_support
# from PIL import Image
# import pycocotools.mask as mask_util

# # Import Detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor, DefaultTrainer
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
# from detectron2.structures import BoxMode
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset


# # Function to find mask files for an image
# def get_masks_for_image(image_filename, mask_dir):
#     mask_prefix = os.path.splitext(image_filename)[0]  # Remove .jpg or .png extension
#     mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith(mask_prefix + "_") and f.endswith(".png")])
#     return [os.path.join(mask_dir, mask_file) for mask_file in mask_files]


# # Convert binary masks into polygons
# def mask_to_polygons(mask):
#     contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     polygons = []
#     for contour in contours:
#         if len(contour) >= 3:
#             polygon = contour.flatten().tolist()
#             if len(polygon) >= 6:
#                 polygons.append(polygon)
#     return polygons if polygons else None


# # Load dataset with mask annotations
# def get_dataset_dicts_with_masks_and_labels(json_path, img_dir, mask_dir):
#     print(f"Loading dataset from {json_path} + {img_dir} + {mask_dir}")
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
#                     "category_id": ann["category_id"],
#                 }

#                 mask_paths = get_masks_for_image(img_info["file_name"], mask_dir)
#                 polygons = []
#                 bitmask_list = []

#                 for mask_path in mask_paths:
#                     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#                     if mask is not None:
#                         mask_polygons = mask_to_polygons(mask)
#                         if mask_polygons:
#                             polygons.extend(mask_polygons)
#                         bitmask_list.append(mask)

#                 if polygons:
#                     obj["segmentation"] = polygons  # Store polygons
#                 elif bitmask_list:  # Convert binary masks to RLE format
#                     rle = mask_util.encode(np.asarray(bitmask_list, order="F"))[0]
#                     obj["segmentation"] = rle

#                 objs.append(obj)

#         record["annotations"] = objs
#         dataset_dicts.append(record)

#     return dataset_dicts

# # Function to evaluate model
# def evaluate_model(cfg, predictor, dataset_name):
#     evaluator = COCOEvaluator(dataset_name, cfg, True, output_dir="./output2")
#     val_loader = build_detection_test_loader(cfg, dataset_name)
    
#     print("🚀 Running model evaluation...")
#     results = inference_on_dataset(predictor.model, val_loader, evaluator)

#     mask_ap_man = results["segm"]["AP-50-person"]
#     mask_ap_woman = results["segm"]["AP-50-woman"]

#     print("\n📊 Evaluation Results:")
#     print(f"🔹 Mask AP (Man): {mask_ap_man:.2f}")
#     print(f"🔹 Mask AP (Woman): {mask_ap_woman:.2f}")

#     return results


# # Function to visualize model predictions
# def visualize_predictions(predictor, dataset_dicts, metadata, num_images=5):
#     for d in random.sample(dataset_dicts, num_images):
#         img = cv2.imread(d["file_name"])
#         outputs = predictor(img)

#         v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)
#         v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

#         plt.figure(figsize=(10, 10))
#         plt.imshow(v.get_image())
#         plt.axis("off")
#         plt.show()


# if __name__ == "__main__":
#     freeze_support()

#     train_json_path = "../t_annotations/instances_Train_updated.json"
#     img_dir_train = "../LV-MHP-v2/train/images"
#     mask_dir_train = "../LV-MHP-v2/train/masks"
#     val_json_path = "../t_annotations/instances_Val_updated.json"
#     img_dir_val = "../LV-MHP-v2/val/images"
#     mask_dir_val = "../LV-MHP-v2/val/masks"

#     DatasetCatalog.register("LVMHPV2_train", lambda: get_dataset_dicts_with_masks_and_labels(train_json_path, img_dir_train, mask_dir_train))
#     DatasetCatalog.register("LVMHPV2_val", lambda: get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val, mask_dir_val))

#     MetadataCatalog.get("LVMHPV2_train").set(thing_classes=["person", "man", "woman", "child"])
#     MetadataCatalog.get("LVMHPV2_val").set(thing_classes=["person", "man", "woman", "child"])

#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
#     cfg.DATASETS.TEST = ("LVMHPV2_val",)
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
#     cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.BASE_LR = 0.0001
#     cfg.SOLVER.MAX_ITER = 150
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.MODEL.DEVICE = "cuda"
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     trainer = DefaultTrainer(cfg)
#     trainer.resume_or_load(resume=False)
#     trainer.train()

#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     predictor = DefaultPredictor(cfg)

#     evaluate_model(cfg, predictor, "LVMHPV2_val")

#     val_dataset_dicts = get_dataset_dicts_with_masks_and_labels(val_json_path, img_dir_val, mask_dir_val)
#     visualize_predictions(predictor, val_dataset_dicts, MetadataCatalog.get("LVMHPV2_val"))