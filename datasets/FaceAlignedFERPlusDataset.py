import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from mtcnn import MTCNN

class FaceAlignedFERPlusDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_csv,
        transform=None,
        aug_transform=None,
        mode="majority",
        nf_threshold=0,
        unknown_threshold=0,
        face_detector="opencv",
        face_cascade_path="haarcascade_frontalface_default.xml",
    ):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            label_csv (str): Path to the label.csv file (no headers).
            transform (callable, optional): Basic transforms (e.g., resize, normalize).
            aug_transform (callable, optional): Augmentations (e.g., random flip).
            mode (str): "majority" or others (currently only 'majority' supported).
            nf_threshold (int): Maximum allowed NF votes for including an image.
            unknown_threshold (int): Maximum allowed Unknown votes for including an image.
            face_detector (str): 'opencv' or 'mtcnn'.
            face_cascade_path (str): Path to a Haar Cascade .xml if using OpenCV.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.aug_transform = aug_transform
        self.mode = mode
        self.nf_threshold = nf_threshold
        self.unknown_threshold = unknown_threshold
        self.face_detector = face_detector.lower()

        # Define CSV columns
        columns = [
            "Image name", "Bounding box", "Neutral", "Happiness", "Surprise",
            "Sadness", "Anger", "Disgust", "Fear", "Contempt", "Unknown", "NF"
        ]
        self.data = pd.read_csv(label_csv, header=None, names=columns)

        # Filter by NF and Unknown thresholds
        self.data = self.data[
            (self.data["NF"] <= nf_threshold) & (self.data["Unknown"] <= unknown_threshold)
        ]

        # Emotion columns
        self.emotion_columns = [
            "Neutral", "Happiness", "Surprise", "Sadness",
            "Anger", "Disgust", "Fear", "Contempt"
        ]

        # If using OpenCV, load Haar Cascade
        if self.face_detector == "opencv":
            if not os.path.exists(face_cascade_path):
                raise FileNotFoundError(
                    f"Haar Cascade file not found at: {face_cascade_path}\n"
                    "Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades\n"
                    "Or update your path if you placed it elsewhere."
                )
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # If using MTCNN, initialize the detector
        elif self.face_detector == "mtcnn":
            self.mtcnn_detector = MTCNN()
        else:
            raise ValueError("face_detector must be 'opencv' or 'mtcnn'")

    def __len__(self):
        return len(self.data)

    def detect_and_crop_face(self, image_pil, desired_size=(48, 48)):
        """
        Detect and crop the first face from the image to desired_size.
        If detection fails, fallback to resizing the entire image.
        """

        # Convert PIL -> NumPy (OpenCV uses BGR, MTCNN typically uses RGB).
        # We'll keep a BGR copy for Haar, an RGB copy for MTCNN.
        image_np = np.array(image_pil)  # shape: (H, W, 3) in RGB
        bgr_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # For Haar
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        if self.face_detector == "opencv":
            faces = self.face_cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Take the first face
                face_crop = bgr_img[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, desired_size)
                cropped_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                return cropped_pil
            else:
                # No face found => resize entire image
                fallback = cv2.resize(bgr_img, desired_size)
                return Image.fromarray(cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB))

        elif self.face_detector == "mtcnn":
            # MTCNN - RGB
            detections = self.mtcnn_detector.detect_faces(image_np)
            if len(detections) > 0:
                # Take the first detection
                # detection['box'] => (x, y, width, height)
                x, y, w, h = detections[0]['box']
                # MTCNN can yield negative x, y if face is near top-left corner, so let's clamp them
                x = max(0, x)
                y = max(0, y)
                face_crop = image_np[y:y+h, x:x+w]  # Still in RGB
                face_crop = cv2.resize(face_crop, desired_size)
                cropped_pil = Image.fromarray(face_crop)  # face_crop is already RGB
                return cropped_pil
            else:
                # No face => fallback
                fallback = cv2.resize(image_np, desired_size)
                return Image.fromarray(fallback)

        # Just in case, fallback:
        fallback = cv2.resize(bgr_img, desired_size)
        return Image.fromarray(cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["Image name"]
        img_path = os.path.join(self.image_dir, img_name)

        image_pil = Image.open(img_path).convert("RGB")

        # Detect and crop
        aligned_image = self.detect_and_crop_face(image_pil, desired_size=(48, 48))

        # Augmentation (50% chance)
        if self.aug_transform and random.random() < 0.5:
            aligned_image = self.aug_transform(aligned_image)
        elif self.transform:
            aligned_image = self.transform(aligned_image)

        # "Majority" label from votes
        votes = row[self.emotion_columns].values.astype(float)
        if self.mode == "majority":
            label_index = votes.argmax()
            label = torch.tensor(label_index, dtype=torch.long)
        else:
            raise ValueError("Only 'majority' mode is implemented.")

        return aligned_image, label
