import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class FERPlusDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform, aug_transform, mode="majority", nf_threshold=0, unknown_threshold=0):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            label_csv (str): Path to the label.csv file (no headers).
            transform (callable, optional): Transform to apply to the images.
            mode (str): "majority", "probabilistic", or "multi_target".
            nf_threshold (int): Maximum allowed NF votes for including an image.
            unknown_threshold (int): Maximum allowed Unknown votes for including an image.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.nf_threshold = nf_threshold
        self.unknown_threshold = unknown_threshold
        self.aug_transform = aug_transform

        # Define column names for the CSV
        columns = [
            "Image name", "Bounding box", "Neutral", "Happiness", "Surprise", 
            "Sadness", "Anger", "Disgust", "Fear", "Contempt", "Unknown", "NF"
        ]

        # Load the CSV without headers and assign column names
        self.data = pd.read_csv(label_csv, header=None, names=columns)

        # Filter out images based on NF and Unknown thresholds
        self.data = self.data[
            (self.data["NF"] <= nf_threshold) & (self.data["Unknown"] <= unknown_threshold)
        ]

        self.emotion_columns = ["Neutral", "Happiness", "Surprise", "Sadness", 
                                "Anger", "Disgust", "Fear", "Contempt"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_name = self.data.iloc[idx, 0]  # First column is the image name
        img_path = f"{self.image_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")
        
        
        if random.random() < 0.5 and self.aug_transform:  # 50% chance to apply augmentation
            image = self.aug_transform(image)
        else:
            image = self.transform(image)

        # Process label
        votes = self.data.iloc[idx, 2:10].values.astype(float)  # Emotion vote counts
        
        if self.mode == "majority":
            label = torch.tensor(votes.argmax(), dtype=torch.long)
        else:
            raise ValueError("Invalid mode. Choose 'majority'.")
        
        return image, label
