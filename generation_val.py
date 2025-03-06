import os
import json
import cv2
from scipy.io import loadmat

class_ids = {
    "person": 0,
    "man": 1,
    "woman": 2,
    "child": 3
}

def generate_val_data_list(dataset_dir):
    images_dir = os.path.join(dataset_dir, "images")
    parsing_annos_dir = os.path.join(dataset_dir, "parsing_annos")
    pose_annos_dir = os.path.join(dataset_dir, "pose_annos")

    data_list = []

    for img_filename in os.listdir(images_dir):
        if not (img_filename.endswith(".jpg") or img_filename.endswith(".png")):
            continue

        print(f"Processing image: {img_filename}")
        img_path = os.path.join(images_dir, img_filename)

        # Read image to get dimensions
        try:
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        # Record structure
        record = {
            "filepath": img_path,
            "width": width,
            "height": height,
            "bboxes": [],
            "keypoints": []
        }

        # Extract image index
        img_index = os.path.splitext(img_filename)[0]

        # Find parsing annotations
        for parsing_filename in os.listdir(parsing_annos_dir):
            if parsing_filename.startswith(img_index):
                parsing_path = os.path.join(parsing_annos_dir, parsing_filename)

                # Extract bounding box from parsing mask
                try:
                    mask = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
                    coords = cv2.findNonZero(mask)
                    x, y, w, h = cv2.boundingRect(coords)
                    x1, y1, x2, y2 = x, y, x + w, y + h

                    record["bboxes"].append({
                        "class": "person",
                        "ann_path": parsing_path,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
                except Exception as e:
                    print(f"Error processing parsing annotation {parsing_path}: {e}")

        # Find pose annotations
        pose_filename = f"{img_index}.mat"
        pose_path = os.path.join(pose_annos_dir, pose_filename)
        if os.path.exists(pose_path):
            try:
                mat_data = loadmat(pose_path)
                keypoints = mat_data.get("keypoints", [])
                if hasattr(keypoints, "tolist"):
                    record["keypoints"] = keypoints.tolist()
                else:
                    record["keypoints"] = keypoints
            except Exception as e:
                print(f"Error reading pose annotation {pose_path}: {e}")

        # Add record to data list
        if record["bboxes"]:
            data_list.append(record)

    # Save to data_list.json
    output_path = os.path.join(dataset_dir, "data_list.json")
    with open(output_path, "w") as json_file:
        json.dump(data_list, json_file, indent=4)
    print(f"Generated {len(data_list)} entries in {output_path}")

# Run the function for validation dataset
generate_val_data_list("C:\\Users\\bader\\Desktop\\Hument_detection\\detectron2\\..\\LV-MHP-v2\\val")