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
from tqdm import tqdm
import random 
import numpy as np
import json
from detectron2.structures import BoxMode

def visualize_predictions(predictor, image_dir, metadata, num_samples=5):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    for img_path in random.sample(image_files, num_samples):
        img = cv2.imread(img_path)
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.numpy()
        
        for i, box in enumerate(instances.pred_boxes):
            class_id = pred_classes[i]
            label = "Man" if class_id == 0 else "Woman" if class_id == 1 else "Unknown"
            v.draw_text(label, box[:2], font_size=10, color="w")

        vis = v.draw_instance_predictions(instances)

        plt.figure(figsize=(10, 10))
        plt.imshow(vis.get_image())
        plt.axis("off")
        plt.title(f"Predictions for {os.path.basename(img_path)}")
        plt.show()

def evaluate_model(predictor, image_dir):
    y_true, y_pred = [], []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    print("Evaluating on test data...")
    for img_path in tqdm(image_files[:1]):  
        img = cv2.imread(img_path)
        outputs = predictor(img)
        
        if "instances" in outputs:
            pred_classes = outputs["instances"].pred_classes.cpu().numpy()
            y_pred.extend(pred_classes)
            y_true.extend([1] * len(pred_classes)) 
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("Gender Classification Metrics:")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

if __name__ == "__main__":
    # Set dataset paths
    test_img_dir = "../LV-MHP-v2/test/images/"  # Test images directory
    
    mhp_metadata = MetadataCatalog.get("mhp_train")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ()  # No annotations for test set
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "./ph/model_final.pth"  # Load trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    
    evaluate_model(predictor, test_img_dir)
    visualize_predictions(predictor, test_img_dir, mhp_metadata)