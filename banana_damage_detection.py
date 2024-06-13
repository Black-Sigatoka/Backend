import cv2
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO
import matplotlib.pyplot as plt

class BananaDamageDetector:
    def __init__(self, model_path):
        self.device = 'cpu'
        print("Using Device: ", self.device)
        self.model = YOLO(model_path)
        print("Model loaded successfully")

    def detect(self, image):
        results = self.model(image, imgsz=640)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
        else:
            masks = None

        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        return boxes, masks, scores, labels, results

    def draw_bounding_boxes(self, image, boxes, masks, scores, labels, banana_severity=None):
        print("Entering draw_bounding_boxes function.")
        
        image_copy = image.copy()

        banana_class = 0  # banana class is 1
        banana_color = (0, 255, 0)  # green color in BGR format
        damage_class = 1  # damage class is 0
        damage_color = (0, 0, 255)  # red color in BGR format

        for i in range(len(labels)):
            box = boxes[i]
            score = scores[i]
            label = labels[i]
            mask = masks[i]

            if label == banana_class:
                color = banana_color
                if banana_severity is not None and i in banana_severity:
                    banana_id = banana_severity[i]["Banana ID"]  # Get banana ID from the dictionary
                    print(f'Banana ID: {banana_id} detected.')
                    text = f'Banana ID: {banana_id} . Score: {score:.2f}'
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)

                    # Draw the text background with yellow color
                    cv2.rectangle(image_copy, (box[0], box[1] - 20), (box[0] + w, box[1]), color, -1)

                    # Draw the text with white color
                    cv2.putText(image_copy, text, (box[0], box[1] - 5), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5,
                                color=(0, 0, 0), thickness=1)

                cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), color, 2)

                # Testing out the mango mask
                if mask is not None:
                    mask_indices = mask == 1  # Get the indices where the mask is active
                    if mask_indices.any():  # Check if there are any active pixels in the mask
                        mask_image = np.zeros(image.shape, dtype=np.uint8)
                        mask_image[mask == 1] = color
                        image_copy[mask == 1] = cv2.addWeighted(image_copy[mask == 1], 0.5, mask_image[mask == 1], 0.5, 0)

            elif label == damage_class:
                color = damage_color
                if mask is not None:
                    mask_indices = mask == 1  # Get the indices where the mask is active
                    if mask_indices.any():  # Check if there are any active pixels in the mask
                        # Create a mask image with the specified color
                        mask_image = np.zeros(image.shape, dtype=np.uint8)
                        mask_image[mask_indices] = color
                        # Apply weighted addition only to regions where the mask is active
                        image_copy[mask_indices] = cv2.addWeighted(image_copy[mask_indices], 0.5,
                                                                    mask_image[mask_indices], 0.5, 0)

        return image_copy
    

    def save_annotated_image(self, image, boxes, masks, scores, labels, output_path):
        image_copy = self.draw_bounding_boxes(image, boxes, masks, scores, labels)
        cv2.imwrite(output_path, image_copy)

    
class BananaDamageAnalyzer:
    def __init__(self, image_path, num_masks, results, class_ids, W, H):
        self.image_path = image_path
        self.num_masks = num_masks
        self.results = results
        self.class_ids = class_ids
        self.W = W
        self.H = H
        self.polygons_damage = []  # List to store the polygons
        self.polygons_banana = []
        self.image = None
        self.tracker_ids = []  # Initialize an empty list for tracker IDs
        self.banana_ids = {}  # Dictionary to store banana IDs by label
        self.threshold = 0.3
        self.detections_damage = []
        self.detections_banana = []

    def load_image(self):
        self.image = cv2.imread(self.image_path)

    def compute_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

        iou = intersection / union if union > 0 else 0
        return iou

    def is_damage_inside_banana(self, damage_box, banana_box):
        """
        Check if the damage bounding box is fully contained within the banana bounding box.
        """
        x1_d, y1_d, x2_d, y2_d = damage_box
        x1_m, y1_m, x2_m, y2_m = banana_box

        # Check that all points of the damage box are within the mango box
        return x1_m <= x1_d and y1_m <= y1_d and x2_m >= x2_d and y2_m >= y2_d

    def assign_tracker_ids(self):
        if self.results is not None:
            for i in range(len(self.class_ids)):
                x = (self.results.masks.xyn[i][:, 0] * self.W).astype("int")
                y = (self.results.masks.xyn[i][:, 1] * self.H).astype("int")
                points = np.column_stack((x, y))

                if i < len(self.results.boxes.xyxy):
                    bbox = self.results.boxes.xyxy[i]
                else:
                    bbox = None

                if i < len(self.results.masks.data):
                    mask = self.results.masks.data[i]
                else:
                    mask = None

                if self.class_ids[i] == 0:
                    self.banana_ids[i] = len(self.banana_ids)  # Assign a unique identifier to each banana
                    self.detections_banana.append(points)
                    if len(points) >= 4:
                        poly1 = Polygon(points)
                        self.polygons_banana.append(poly1)
                    else:
                        self.polygons_banana.append(None)

                    if bbox is not None and mask is not None:
                        self.tracker_ids.append({'banana_id': i, 'damage_id': None})
                else:
                    self.detections_damage.append(points)
                    if len(points) >= 4:
                        poly2 = Polygon(points)
                        self.polygons_damage.append(poly2)
                    else:
                        self.polygons_damage.append(None)

                    if bbox is not None and mask is not None:
                        self.tracker_ids.append({'banana_id': None, 'damage_id': i})

    def analyze_damage(self):
        banana_to_damage = {}  # Dictionary to map banana to their assigned damages

        for i, banana in enumerate(self.detections_banana):
            banana_to_damage[i] = []  # Initialize an empty list of assigned damages for each banana

        # Create a list to track whether each damage has been assigned
        damage_assigned = [False] * len(self.detections_damage)

        for i, banana in enumerate(self.detections_banana):
            for j, damage in enumerate(self.detections_damage):
                if not damage_assigned[j] and self.is_damage_inside_banana(self.results.boxes.xyxy[j], self.results.boxes.xyxy[i]):
                    banana_to_damage[i].append(j)  # Assign this damage to the banana
                    damage_assigned[j] = True  # Mark this damage as assigned

        return banana_to_damage

    def compute_severity(self, damage_list):
        total_damage_area = 0
        banana_area = 0

        for damage_idx in damage_list:
            if 0 <= damage_idx < len(self.polygons_damage) and self.polygons_damage[damage_idx] is not None and len(self.polygons_damage[damage_idx].exterior.coords) >= 4:
                total_damage_area += self.polygons_damage[damage_idx].area

        for banana_idx in damage_list:
            if 0 <= banana_idx < len(self.polygons_banana) and self.polygons_banana[banana_idx] is not None and len(self.polygons_banana[banana_idx].exterior.coords) >= 4:
                banana_area += self.polygons_banana[banana_idx].area

        if banana_area == 0 and total_damage_area > 0:
            severity_level = 1.0  # If banana area is zero and there's damage, consider the severity level as 100%
        elif banana_area > 0:
            severity_level = total_damage_area / (banana_area + total_damage_area)
        else:
            severity_level = 0.0


        return total_damage_area, banana_area, severity_level

    def visualize_polygons(self):
        if self.image is not None:
            for i, poly in enumerate(self.polygons_damage):
                if poly is not None:
                    plt.plot(*poly.exterior.xy, color='red', alpha=0.5)

            for i, poly in enumerate(self.polygons_banana):
                if poly is not None:
                    plt.plot(*poly.exterior.xy, color='green', alpha=0.5)
            plt.imshow(self.image)