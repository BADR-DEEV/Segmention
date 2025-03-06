from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
import os
import json
import random
from PIL import Image, ImageFile
import cv2

# Import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from multiprocessing import freeze_support

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Class mappings
class_ids = {
    "person": 0,
    "man": 1,
    "woman": 2,
    "child": 3
}

def get_LVMHPV2_dicts(img_dir):
    json_file = os.path.join(img_dir, "data_list.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"data_list.json not found in {img_dir}")

    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        try:
            record = {
                "file_name": os.path.normpath(v["filepath"]),
                "image_id": idx,
                "height": v["height"],
                "width": v["width"],
                "annotations": [
                    {
                        "bbox": [box["x1"], box["y1"], box["x2"], box["y2"]],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": class_ids.get(box["class"], -1),
                    }
                    for box in v["bboxes"]
                ],
            }
            # Filter invalid category IDs
            record["annotations"] = [anno for anno in record["annotations"] if anno["category_id"] != -1]
            dataset_dicts.append(record)
        except KeyError as e:
            print(f"Skipping record due to missing key: {e}")
    return dataset_dicts


if __name__ == '__main__':
    print("Starting evaluation...")
    freeze_support()

    # Register datasets
    for d in ["train", "val"]:
        DatasetCatalog.register(f"LVMHPV2_{d}", lambda d=d: get_LVMHPV2_dicts(f"../LV-MHP-v2/{d}"))
        MetadataCatalog.get(f"LVMHPV2_{d}").set(thing_classes=["person", "man", "woman", "child"])

    LVMHP_meta = MetadataCatalog.get("LVMHPV2_train")

    # Set up the model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("LVMHPV2_val",)
    cfg.MODEL.DEVICE = "cuda"

    # Initialize the predictor
    predictor = DefaultPredictor(cfg)

    # Perform evaluation
    evaluator = COCOEvaluator("LVMHPV2_val", cfg, False, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "LVMHPV2_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Print evaluation results
    print("\nEvaluation Results:")
    print(results)

    # Visualization of predictions
    val_dataset_dicts = get_LVMHPV2_dicts("../LV-MHP-v2/val")
    for d in random.sample(val_dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)


        v = Visualizer(
            img[:, :, ::-1],
            metadata=LVMHP_meta,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,
        )
        try:
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            print("Image visualized.")

        except Exception as e:
            print(f"Error visualizing image: {e}")



