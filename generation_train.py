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
        print(f"waiting")
        print(f"Processing image: {img_filename}")
        img_path = os.path.join(images_dir, img_filename)
        print(f"waiting")


        # Read image to get dimensions
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to read image {img_path}. Skipping.")
                continue
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
        matching_parsing_files = [f for f in os.listdir(parsing_annos_dir) if f.startswith(img_index)]
        if not matching_parsing_files:
            print(f"No matching parsing annotations for {img_index}")
            continue

        for parsing_filename in matching_parsing_files:
            parsing_path = os.path.join(parsing_annos_dir, parsing_filename)

            try:
                mask = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Error: Unable to read mask {parsing_path}.")
                    continue
                coords = cv2.findNonZero(mask)
                
                if coords is None:
                    print(f"Warning: No non-zero pixels in {parsing_path}. Skipping.")
                    continue
                x, y, w, h = cv2.boundingRect(coords)
                record["bboxes"].append({
                    "class": "person",
                    "ann_path": parsing_path,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                })
                print(f"Added bbox from {parsing_path}")
            except Exception as e:
                print(f"Error processing mask {parsing_path}: {e}")

        # Find pose annotations
        pose_path = os.path.join(pose_annos_dir, f"{img_index}.mat")
        if not os.path.exists(pose_path):
            print(f"No pose annotation found for {img_index}.")
        else:
            try:
                mat_data = loadmat(pose_path)
                keypoints = mat_data.get("keypoints", [])
                record["keypoints"] = keypoints.tolist() if hasattr(keypoints, "tolist") else keypoints
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
generate_val_data_list("C:\\Users\\bader\\Desktop\\Hument_detection\\detectron2\\..\\LV-MHP-v2\\train")