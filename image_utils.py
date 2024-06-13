import cv2
import numpy as np
import urllib.request

def read_image_from_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    
    if img is None:
        print(f"Error reading image from URL: {url}")
        return None

    return img