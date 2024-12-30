import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time
from CNN import CNN


model_path = "models/model_>80.pth"  # Adjust to your actual path
device = torch.device("mps")  # Change to "cuda" if you have a CUDA GPU

# If used `torch.save(model, ...)`, ensure CNN is imported or defined in scope
model = torch.load(model_path, map_location=device)
model.eval()
model.to(device)



cascade_path = "datasets/haarcascade_frontalface_default.xml"  # Path to the Haar Cascade XML
face_cascade = cv2.CascadeClassifier(cascade_path)


# The same transforms used at training time (minus heavy augmentation).
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Emotion labels in the order your model was trained
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


def get_top_emotion_probs(logits, top_k=3):
    """
    Return the top_k emotions with their probabilities (sorted descending).
    """
    softmax = torch.nn.functional.softmax(logits, dim=1)  # shape: (1, num_emotions)
    probs = softmax.squeeze().cpu().numpy()  # shape: (num_emotions,)

    sorted_indices = np.argsort(probs)[::-1]  # descending
    top_results = []
    for i in range(top_k):
        idx = sorted_indices[i]
        top_results.append((emotion_labels[idx], probs[idx]))
    return top_results


###############################
# Video Capture Config
###############################
cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the live stream.")

# FPS measurement
prev_time = time.time()
frame_count = 0

# If you only want the largest face
ONLY_LARGEST_FACE = False

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

    # Optionally only detect the largest face
    if ONLY_LARGEST_FACE and len(faces) > 1:
        areas = [(w * h) for (x, y, w, h) in faces]
        max_idx = np.argmax(areas)
        faces = [faces[max_idx]]

    # For each face, perform prediction
    for (x, y, w, h) in faces:
        face_region = frame[y:y + h, x:x + w]

        # Convert to PIL for transforms
        face_pil = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)  # (1, 3, 48, 48)

        with torch.no_grad():
            outputs = model(face_tensor)  # shape: (1, num_emotions)
            _, predicted = torch.max(outputs, 1)
            emotion_index = predicted.item()
            emotion_text = emotion_labels[emotion_index]

            # Get top-3 probabilities for a more detailed display
            top_probs = get_top_emotion_probs(outputs, top_k=3)

        # Draw bounding box
        color = (255, 0, 0)  # Blue-ish
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(frame, emotion_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display top-3 probabilities below the face
        offset_y = 20
        for idx, (lbl, prob) in enumerate(top_probs):
            text = f"{lbl}: {prob * 100:.1f}%"
            cv2.putText(frame, text, (x, y + h + 20 + idx * offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    cv2.imshow("Live Emotion Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
