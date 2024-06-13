import os
import json
import numpy as np
import cv2
import logging
import base64

from azureml.core.model import Model
from banana_damage_detection import BananaDamageDetector, BananaDamageAnalyzer
from image_utils import read_image_from_url 


# Called when the service is loaded
def init():
    global detector

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "best.pt"
    )

    # Initialize the BananaDamageDetector with the model
    detector = BananaDamageDetector(model_path)
    logging.info("Init complete")

def run(input_data):
    try:
        # Load the image data
        input_data = json.loads(input_data)
        image_paths= input_data.get("image_urls")

        if image_paths is None:
            raise ValueError("Input data does not contain 'images' key.")
        
        # Dictionary to store banana severity information
        banana_severity = {}

        # Perform inference for each image
        post_results = []
        
        for image_path in image_paths:
            image = read_image_from_url(image_path)

            if image is None:
                post_results.append({"error": f"Failed to load image from URL: {image_path}"})
                continue

            image = cv2.resize(image, (640, 640))
            H, W, _ = image.shape

            boxes, masks, scores, labels, results = detector.detect(image)

            if results is not None:  # Check if there are detections
                num_detections = len(results.boxes)  # Get the number of detections
                analyzer = BananaDamageAnalyzer(image_path, num_detections, results, labels, W, H)
                analyzer.assign_tracker_ids()

                for banana_id, detection in analyzer.analyze_damage().items():
                    total_damage_area, banana_area, severity_level = analyzer.compute_severity(detection)
                    
                    # Determine severity level
                    if severity_level == 0:
                        level = "No Damage"
                    elif severity_level < 0.1:
                        level = "Low"
                    elif severity_level < 0.3:
                        level = "Intermediate"
                    else:
                        level = "High"

                    # Store mango severity information by mango ID
                    banana_severity_info = {
                        "Total Sigatoka Damage Area": total_damage_area,
                        "Banana Leaf Area": banana_area,
                        "Severity Level": severity_level * 100,
                        "Severity Level Category": level,
                    }

                    # Draw the bounding boxes
                    image_copy = detector.draw_bounding_boxes(image, boxes, masks, scores, labels, banana_severity_info)
                    _, buffer = cv2.imencode('.jpg', image_copy)
                    image_str = base64.b64encode(buffer).decode('utf-8')
                    banana_severity_info["image_copy"] = image_str

                    post_results.append(banana_severity_info)
            else:
                post_results.append({"error": "No detections found."})

        # Return the results as JSON
        return json.dumps(post_results)
        
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
