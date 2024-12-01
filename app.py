from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import cv2
import numpy as np
from flask_cors import CORS
import os
import mediapipe as mp
app = Flask(__name__)
CORS(app)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
base_image_path = r"spin1.png"
base_image = cv2.imread(base_image_path)
@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        # Receive the base64 image from the frontend
        data = request.json
        image_base64 = data.get("image")
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        # Decode the base64 image
        image_data = base64.b64decode(image_base64.split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Crop faces and return the first detected face
        cropped_face = None
        for (x, y, w, h) in faces:
            # Crop the face from the image without drawing any lines
            cropped_face = image_np[y:y + h, x:x + w]
            break  # Only process the first face detected, remove if you want multiple faces

        if cropped_face is not None:
            # Convert the cropped face back to the correct format (RGB)
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)


            # Create an elliptical mask for the face
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            axes = (w // 2, h // 2)  # axes for the ellipse (width and height)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

#         # Apply the mask to the cropped face (making it oval)
            face_with_mask = cv2.bitwise_and(cropped_face, cropped_face, mask=mask)

#         # Convert face with mask to an image with transparency (4 channels: BGRA)
            face_with_mask_bgra = cv2.cvtColor(face_with_mask, cv2.COLOR_BGR2BGRA)
            face_with_mask_bgra[:, :, 3] = mask  # Set the alpha channel (transparency) based on the mask

#         # Convert the base image to HSV for better color detection
            hsv_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)

#         # Define the green color range in HSV space
            lower_green = np.array([35, 50, 50])  # lower bound of green
            upper_green = np.array([85, 255, 255])  # upper bound of green

#         # Threshold the image to get the green region
            mask_green = cv2.inRange(hsv_base, lower_green, upper_green)

#         # Find the contours of the green spots
            contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the bounding box of the largest green spot
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Resize the cropped face with the elliptical mask to fit the green region
            resized_face = cv2.resize(face_with_mask_bgra, (w, h))

            # Make a fresh copy of the base image before modifying it
            base_image_copy = base_image.copy()

            # Replace the green region with the resized face (keeping the oval shape)
            for i in range(h):
                for j in range(w):
                    # Get the pixel of the resized face and the alpha value (transparency)
                    if resized_face[i, j, 3] > 0:  # Only overwrite non-transparent pixels
                        base_image_copy[y + i, x + j] = resized_face[i, j, :3]  # Replace the RGB value
            # Convert the processed image (cropped face) to base64
            _, buffer = cv2.imencode(".png", base_image_copy)
            processed_image_base64 = base64.b64encode(buffer).decode("utf-8")

            # Send the processed image back to the frontend
            return jsonify({"processed_image": processed_image_base64})
        else:
            return jsonify({"error": "No face detected"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
