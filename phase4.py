# import os
# import cv2
# import json
# import random
# import torch
# import numpy as np
# import detectron2
# from detectron2.engine import DefaultTrainer, DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.data.datasets import register_coco_instances
# from detectron2 import model_zoo
# from multiprocessing import freeze_support
# # Set dataset paths

# if __name__ == "__main__":
#     # Register dataset
#     freeze_support()
#     data_dir = "../t_annotations/"
#     train_json = os.path.join(data_dir, "full_instances_train_updated.json")
#     val_json = os.path.join(data_dir, "full_instances_val_updated.json")
#     train_img_dir = "../LV-MHP-v2/train/images/"
#     val_img_dir = "../LV-MHP-v2/val/images/"
#     register_coco_instances("mhp_train", {}, train_json, train_img_dir)
#     register_coco_instances("mhp_val", {}, val_json, val_img_dir)

#     # Load Metadata
#     mhp_metadata = MetadataCatalog.get("mhp_train")

#     # Define Model Configuration
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.DATASETS.TRAIN = ("mhp_train",)
#     cfg.DATASETS.TEST = ("mhp_val",)
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.BASE_LR = 0.005
#     cfg.SOLVER.MAX_ITER = 200
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 0: Male, 1: Female
#     cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     cfg.MASK_ON = True

#     # Train Model
#     trainer = DefaultTrainer(cfg)
#     trainer.resume_or_load(resume=True)
#     trainer.train()

#     # Evaluate Model
#     evaluator = COCOEvaluator("mhp_val", cfg, False, output_dir="./output/")
#     val_loader = build_detection_test_loader(cfg, "mhp_val")
#     inference_on_dataset(trainer.model, val_loader, evaluator)

#     # Load Predictor for Inference
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#     predictor = DefaultPredictor(cfg)
#     val_loader = build_detection_test_loader(cfg, "mhp_val")
#     print(inference_on_dataset(predictor.model, val_loader, evaluator))
    
#     def apply_mask(image, mask):
#         """Applies a red mask overlay to the detected female object."""
#         mask = mask.astype(bool)
#         red_overlay = np.zeros_like(image)
#         red_overlay[:, :, 2] = 255  # Set red channel to max
#         image[mask] = cv2.addWeighted(image, 0.5, red_overlay, 0.5, 0)[mask]
#         return image

#     # Run Inference on Validation Images
#     val_images = os.listdir(val_img_dir)
#     random.shuffle(val_images)
#     for img_name in val_images[:5]:
#         img_path = os.path.join(val_img_dir, img_name)
#         image = cv2.imread(img_path)
#         outputs = predictor(image)
        
#         # Extract predictions
#         instances = outputs["instances"]
#         pred_classes = instances.pred_classes.cpu().numpy()
#         masks = instances.pred_masks.cpu().numpy()
        
#         # Apply masks to detected females
#         for i, cls in enumerate(pred_classes):
#             if cls == 1:  # Female class
#                 image = apply_mask(image, masks[i])
        
#         # Visualize results
#         v = Visualizer(image[:, :, ::-1], metadata=mhp_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
#         out = v.draw_instance_predictions(instances.to("cpu"))
#         cv2.imshow("Output", out.get_image()[:, :, ::-1])
#         cv2.waitKey(0)
#     cv2.destroyAllWindows()






import os
import cv2
import json
import random
import torch
import numpy as np
import detectron2
import detectron2.data
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils



def visualize_dataset(dataset_name, metadata, num_samples=5):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("Train Image with Mask", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_predictions(predictor, dataset_name, metadata, num_samples=5):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Predicted Image with Mask", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()



class AugmentedTrainer(DefaultTrainer):
    @classmethod 
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice"),
            T.RandomFlip(),
            T.RandomRotation([0, 90, 180, 270]),
        ])
        return detectron2.data.build_detection_train_loader(cfg, mapper=mapper)

 
def visualize_predictions(predictor, dataset_dicts, metadata):
    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(10, 10))
        plt.imshow(v.get_image())
        plt.axis("off")
        plt.show()

# Set dataset paths
data_dir = "../t_annotations/"
train_json = os.path.join(data_dir, "barka_train.json")
val_json = os.path.join(data_dir, "full_instances_val_updated.json")
train_img_dir = "../LV-MHP-v2/train/images/"
val_img_dir = "../LV-MHP-v2/val/images/"
train_mask_dir = "../LV-MHP-v2/train/masks/"
val_mask_dir = "../LV-MHP-v2/val/masks/"


    
def load_coco_json(json_file, img_dir):
    with open(json_file) as f:
        dataset = json.load(f)
    
    dataset_dicts = []
    for img in dataset["images"]:
        record = {}
        record["file_name"] = os.path.join(img_dir, img["file_name"])
        record["image_id"] = img["id"]
        record["height"] = img["height"]
        record["width"] = img["width"]
        
        annos = [anno for anno in dataset["annotations"] if anno["image_id"] == img["id"]]
        objs = []

        for anno in annos:
            poly = anno["segmentation"]
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": poly , 
                "segmentation": poly if anno["category_id"] == 1 else None, 
                "category_id":  anno["category_id"] 
            }
            objs.append(obj)

            
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == "__main__":
    freeze_support()
    # Register datasets
    DatasetCatalog.register("mhp_train", lambda: load_coco_json(train_json, train_img_dir))
    DatasetCatalog.register("mhp_val", lambda: load_coco_json(val_json, val_img_dir))
    MetadataCatalog.get("mhp_train").set(thing_classes=["man", "woman"])
    MetadataCatalog.get("mhp_val").set(thing_classes=["man", "woman"])

    # Load Metadata
    mhp_metadata = MetadataCatalog.get("mhp_train")

    # Define Model Configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("mhp_train",)
    cfg.DATASETS.TEST = ("mhp_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR =  0.0001
    cfg.SOLVER.MAX_ITER = 2100
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    #THRESH TEST
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 0: Male, 1: Female
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MASK_ON = True
    cfg.OUTPUT_DIR = "./ph"
    cfg.INPUT.MASK_FORMAT = "polygon"
    
    # cfg.INPUT.RANDOM_FLIP = "horizontal"
    # cfg.MODEL.ROI_HEADS.DROPOUT = 0.1  
    # cfg.SOLVER.GAMMA = 0.1  

    # Train Model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # Evaluate Model
    evaluator = COCOEvaluator("mhp_val", cfg, output_dir="./ph")
    val_loader = build_detection_test_loader(cfg, "mhp_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    # Load Predictor for Inference
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("mhp_val", output_dir="./ph")
    val_loader = build_detection_test_loader(cfg, "mhp_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.5
    # val_loader = build_detection_test_loader(cfg, "mhp_val")

    # print(inference_on_dataset(predictor.model, val_loader, evaluator))
    
    # visualize_predictions(predictor, load_coco_json(val_json, val_img_dir), MetadataCatalog.get("mhp_val"))


    # visualize_dataset("mhp_train", mhp_metadata, num_samples=5)

    dataset_dicts = DatasetCatalog.get("mhp_val")
    visualize_predictions(predictor, dataset_dicts, mhp_metadata)
