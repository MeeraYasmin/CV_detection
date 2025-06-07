import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

def load_image_from_url(url):
    try:
        # Add a custom User-Agent header
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        image = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def detect_image(template_url, target_url):
    # Load the template image (the smaller image to search for)
    template = load_image_from_url(template_url)
    if template is None:
        print("Error: Unable to load the template image.")
        return

    # Load the target image (the larger image to search within)
    target = load_image_from_url(target_url)
    if target is None:
        print("Error: Unable to load the target image.")
        return

    # Debugging: Print image shapes
    print(f"Template shape: {template.shape if template is not None else 'None'}")
    print(f"Target shape: {target.shape if target is not None else 'None'}")

    # Perform template matching
    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)

    # Set a threshold for detection
    threshold = 0.5  # Lowered threshold for better detection
    locations = np.where(result >= threshold)

    # Check if any matches are found
    if len(locations[0]) > 0:
        print("Template image is present in the target image.")
        # Draw rectangles around matches
        for pt in zip(*locations[::-1]):
            cv2.rectangle(target, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (255, 0, 0), 2)
        # Display the result
        cv2.imshow("Detected Matches", target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Template image is NOT present in the target image.")

# URLs to the images
template_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/800px-PNG_transparency_demonstration_1.png"
target_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/800px-PNG_transparency_demonstration_1.png"

# Run the detection
detect_image(template_url, target_url)
