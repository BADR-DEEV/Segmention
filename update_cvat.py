# import os
# import cv2
# import json
# import numpy as np
# from tqdm import tqdm

# def get_masks_for_image(image_filename, mask_dir):
#     """Find all mask files corresponding to an image filename."""
#     mask_prefix = os.path.splitext(image_filename)[0]  # Remove .jpg/.png extension
#     mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith(mask_prefix + "_") and f.endswith(".png")])
#     return [os.path.join(mask_dir, mask_file) for mask_file in mask_files]

# def mask_to_bitmask(mask, height, width):
#     """
#     Convert a grayscale mask to a binary bitmask (0s and 1s) and resize it properly.
#     """
#     # Ensure binary format (0 or 1)
#     bitmask = (mask > 0).astype(np.uint8)

#     # Resize if needed
#     if bitmask.shape != (height, width):
#         bitmask = cv2.resize(bitmask, (width, height), interpolation=cv2.INTER_NEAREST)

#     return bitmask

# def update_json_with_bitmasks(json_path, mask_dir, output_json):
#     """Update a COCO-style JSON annotation file with bitmasks & fix category IDs."""
#     print(f"Updating dataset: {json_path} with masks from {mask_dir}")

#     with open(json_path, "r") as f:
#         dataset = json.load(f)

#     # Map image_id to file_name and dimensions
#     image_info = {img["id"]: (img["file_name"], img["height"], img["width"]) for img in dataset["images"]}

#     updated_annotations = []
#     total_masks_found = 0
#     empty_masks_count = 0

#     for ann in tqdm(dataset["annotations"], desc="Processing annotations"):
#         image_id = ann["image_id"]
#         image_filename, height, width = image_info.get(image_id, (None, None, None))

#         if not image_filename:
#             print(f"Warning: No filename found for image_id {image_id}. Skipping.")
#             updated_annotations.append(ann)
#             continue

#         # 🔹 Fix category_id (1 → 0, 2 → 1)
#         ann["category_id"] -= 1  

#         # If segmentation is empty, extract from mask
#         if not ann["segmentation"]:
#             mask_paths = get_masks_for_image(image_filename, mask_dir)
#             combined_mask = np.zeros((height, width), dtype=np.uint8)

#             for mask_path in mask_paths:
#                 mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#                 if mask is not None:
#                     bitmask = mask_to_bitmask(mask, height, width) 
#                     combined_mask = np.maximum(combined_mask, bitmask) # Merge multiple masks
#         if np.any(combined_mask):
#             ann["segmentation"] = [combined_mask.tolist()]  # ✅ Convert NumPy array to list before saving
#             total_masks_found += 1
#         else:
#             empty_masks_count += 1

#     updated_annotations.append(ann)

#     dataset["annotations"] = updated_annotations

#     # Save updated JSON
#     with open(output_json, "w") as f:
#         json.dump(dataset, f, indent=4)

#     print(f"✅ Updated {json_path}. Saved as {output_json}")
#     print(f"🎯 Total masks assigned: {total_masks_found}")
#     if empty_masks_count > 0:
#         print(f"⚠️ {empty_masks_count} annotations had no masks!")


# update_json_with_bitmasks("../t_annotations/full_instances_default.json", "../LV-MHP-v2/val/masks", "../t_annotations/full_instances_Val_updated.json")









# import os
# import cv2
# import json
# import numpy as np
# from tqdm import tqdm
# from PIL import Image

# def get_masks_for_image(image_filename, mask_dir):
#     """
#     Find all mask files corresponding to an image filename (not image_id).
#     """
#     mask_prefix = os.path.splitext(image_filename)[0]  # Remove .jpg or .png extension
#     mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith(mask_prefix + "_") and f.endswith(".png")])
#     return [os.path.join(mask_dir, mask_file) for mask_file in mask_files]

# def mask_to_polygons(mask):
#     """
#     Convert a binary mask into Detectron2-compatible polygon format with simplified contours.
#     """
#     contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     polygons = []
    
#     for contour in contours:
#         if len(contour) >= 3:  # Ensure valid polygons
#             # ✅ Simplify the contour (Reduce number of points)
#             epsilon = 0.005 * cv2.arcLength(contour, True)
#             simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
#             polygon = simplified_contour.flatten().tolist()
#             if len(polygon) >= 6:  # Ensure it forms a valid polygon
#                 polygons.append(polygon)
    
#     return polygons if polygons else None

# def update_json_with_masks(json_path, mask_dir, output_json):
#     """
#     Updates a COCO-style JSON annotation file by filling in missing segmentation fields using masks.
#     """
#     print(f"Updating dataset: {json_path} with masks from {mask_dir}")

#     with open(json_path, "r") as f:
#         dataset = json.load(f)

#     # Map image_id to file_name for easy lookup
#     image_id_to_filename = {img["id"]: img["file_name"] for img in dataset["images"]}

#     updated_annotations = []
#     empty_masks_count = 0
#     total_masks_found = 0

#     for ann in tqdm(dataset["annotations"], desc="Processing annotations"):
#         image_id = ann["image_id"]
#         image_filename = image_id_to_filename.get(image_id, None)
        
#         if not image_filename:
#             print(f"Warning: No filename found for image_id {image_id}. Skipping.")
#             updated_annotations.append(ann)
#             continue

#         # If segmentation is empty, try to extract it from the corresponding mask images
#         if not ann["segmentation"]:
#             mask_paths = get_masks_for_image(image_filename, mask_dir)
#             polygons = []

#             for mask_path in mask_paths:
#                 mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#                 if mask is not None:
#                     mask_polygons = mask_to_polygons(mask)
#                     if mask_polygons:
#                         polygons.extend(mask_polygons)

#             if polygons:
#                 ann["segmentation"] = polygons
#                 total_masks_found += 1
#             else:
#                 empty_masks_count += 1

#         updated_annotations.append(ann)

#     dataset["annotations"] = updated_annotations

#     # Save the updated JSON file
#     with open(output_json, "w") as f:
#         json.dump(dataset, f, indent=4)

#     print(f"✅ Finished updating {json_path}.")
#     print(f"🚀 Saved updated JSON as: {output_json}")
#     print(f"🎯 Total masks found and assigned: {total_masks_found}")
#     if empty_masks_count > 0:
#         print(f"⚠️ {empty_masks_count} annotations had no masks found!")

# # Paths for Train and Validation datasets
# train_json_path = "../t_annotations/full_instances_Train.json"
# val_json_path = "../t_annotations/full_instances_default.json"
# train_mask_dir = "../LV-MHP-v2/train/masks"
# val_mask_dir = "../LV-MHP-v2/val/masks"
# updated_train_json = "../t_annotations/full_instances_Train_updated.json"
# updated_val_json = "../t_annotations/full_instances_Val_updated.json"

# # Update JSON for Train and Val datasets
# update_json_with_masks(train_json_path, train_mask_dir, updated_train_json)
# update_json_with_masks(val_json_path, val_mask_dir, updated_val_json)
















import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils  # COCO's RLE encoding
from PIL import Image

def get_masks_for_image(image_filename, mask_dir):
    """
    Find all mask files corresponding to an image filename.
    """
    mask_prefix = os.path.splitext(image_filename)[0]  # Remove .jpg/.png extension
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith(mask_prefix + "_") and f.endswith(".png")])
    return [os.path.join(mask_dir, mask_file) for mask_file in mask_files]

def mask_to_polygons(mask):
    """
    Convert a binary mask into COCO-compatible polygon format.
    """
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        if len(contour) >= 3:  # Ensure valid polygon
            epsilon = 0.002 * cv2.arcLength(contour, True)  # Reduce points slightly
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            polygon = simplified_contour.flatten().tolist()
            if len(polygon) >= 6:  # Ensure at least 3 points (x, y) pairs
                polygons.append(polygon)
    
    return polygons if polygons else None

def mask_to_rle(mask):
    """
    Convert a binary mask to COCO-style RLE (Run-Length Encoding).
    """
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # Convert bytes to string for JSON compatibility
    return rle

def update_json_with_masks(json_path, mask_dir, output_json, use_rle=False):
    """
    Updates a COCO-style JSON annotation file with segmentation masks.
    Supports both polygon and RLE formats.
    """
    print(f"Updating dataset: {json_path} with masks from {mask_dir}")

    with open(json_path, "r") as f:
        dataset = json.load(f)

    image_id_to_filename = {img["id"]: img["file_name"] for img in dataset["images"]}

    updated_annotations = []
    empty_masks_count = 0
    total_masks_found = 0

    for ann in tqdm(dataset["annotations"], desc="Processing annotations"):
        image_id = ann["image_id"]
        image_filename = image_id_to_filename.get(image_id, None)

        if not image_filename:
            print(f"Warning: No filename for image_id {image_id}. Skipping.")
            updated_annotations.append(ann)
            continue

        # Convert category ID to 0-based index
        ann["category_id"] -= 1  

        # If segmentation is empty, extract from mask images
        if not ann["segmentation"]:
            mask_paths = get_masks_for_image(image_filename, mask_dir)
            combined_mask = None

            for mask_path in mask_paths:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = np.maximum(combined_mask, mask)  # Merge multiple masks

            if combined_mask is not None:
                if use_rle:
                    ann["segmentation"] = mask_to_rle(combined_mask)  # Convert to RLE
                else:
                    polygons = mask_to_polygons(combined_mask)
                    if polygons:
                        ann["segmentation"] = polygons
                    else:
                        empty_masks_count += 1
            else:
                empty_masks_count += 1

        updated_annotations.append(ann)

    dataset["annotations"] = updated_annotations

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"✅ Updated {json_path} -> Saved as {output_json}")
    print(f"🎯 Total masks found: {total_masks_found}")
    if empty_masks_count > 0:
        print(f"⚠️ {empty_masks_count} annotations had no masks found!")

# Paths for Train and Validation datasets
train_json_path = "../t_annotations/barka.json"
val_json_path = "../t_annotations/instances_val.json"
train_mask_dir = "../LV-MHP-v2/train/masks"
val_mask_dir = "../LV-MHP-v2/val/masks"
updated_train_json = "../t_annotations/barka_train.json"
updated_val_json = "../t_annotations/barka_val.json"

update_json_with_masks(train_json_path, train_mask_dir, updated_train_json, use_rle=False)  # Use Polygons
update_json_with_masks(val_json_path, val_mask_dir, updated_val_json, use_rle=False)  # Use Polygons