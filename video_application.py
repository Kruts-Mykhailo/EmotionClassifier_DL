import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from CNN import CNN

###############################
# 1) Load Your Trained Model
###############################

# Make sure this matches your saved model's architecture
# e.g., if you have a class CNN with a certain constructor signature
# from your_code_file import CNN  # or define your CNN architecture in this file

# Example: we load a saved model
model_path = "models/model_cnn.pth"  # Adjust to your actual path
device = torch.device("mps")

# #If you have the CNN class defined, do something like:
model = CNN(dropout_rate=0.4, num_emotions=8).to(device)
model.load_state_dict(torch.load("models/model_cnn.pth"))
model.eval()

# # Alternatively, if you saved the entire model via torch.save(model, ...):
# model = torch.load(model_path, map_location=device)
# model.eval()
# model.to(device)

###############################
# 2) Set Up Face Detector (Haar Cascade)
###############################
cascade_path = "datasets/haarcascade_frontalface_default.xml"  # Download if needed
face_cascade = cv2.CascadeClassifier(cascade_path)

###############################
# 3) Define Preprocessing
###############################
# The same transformations used during training (minus any heavy augmentation)
transform = transforms.Compose([
    transforms.Resize((48, 48)),    # Safety resize, in case the face crop is not exact
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Emotion labels (in the order your model was trained)
emotion_labels = [
    "Neutral",
    "Happiness",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
    "Contempt"
]

###############################
# 4) Capture Video Stream
###############################
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the live stream.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Crop the face region
        face_region = frame[y:y+h, x:x+w]

        # Convert to PIL for transforms
        face_pil = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil)           # Apply same transforms as training
        face_tensor = face_tensor.unsqueeze(0).to(device)  # Shape: (1, 3, 48, 48)

        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion_index = predicted.item()
            emotion_text = emotion_labels[emotion_index]

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Put label text above the box
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Live Emotion Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
