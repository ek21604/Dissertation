import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm

# Paths to validation and testing folders
base_dir = r"C:\Users\price\Documents\Uni\Dis\F1 Highlight Videos\Images"
folders = ["Validation", "Testing"]

# Motion Blur Function
def random_motion_blur(img):
    if random.random() > 0.5:
        size = random.choice([3, 5, 7])  # Kernel size (larger = stronger blur)
        angle = random.randint(0, 360)  # Random blur direction

        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[(size - 1) // 2, :] = np.ones(size)
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
        kernel /= size

        # Apply blur
        img = cv2.filter2D(np.array(img), -1, kernel)
        return Image.fromarray(img)
    
    return img

# Other Augmentation Functions
def random_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)

def random_contrast(img):
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)

def random_blur(img):
    if random.random() > 0.5:
        img = cv2.GaussianBlur(np.array(img), (5, 5), sigmaX=1.5)
        return Image.fromarray(img)
    return img

def random_rotation(img):
    return img.rotate(random.choice([-10, -5, 0, 5, 10]))

def random_horizontal_flip(img):
    if random.random() > 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

# Apply augmentations and save new images
for folder in folders:
    for team in os.listdir(os.path.join(base_dir, folder)):
        team_folder = os.path.join(base_dir, folder, team)
        if not os.path.isdir(team_folder):
            continue

        print(f"Augmenting images in: {team_folder}")
        images = [f for f in os.listdir(team_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in tqdm(images, desc=f"Processing {team}"):
            img_path = os.path.join(team_folder, img_name)
            img = Image.open(img_path).convert("RGB")

            # Apply augmentations
            aug_img = random_brightness(img)
            aug_img = random_contrast(aug_img)
            aug_img = random_blur(aug_img)
            aug_img = random_motion_blur(aug_img)  
            aug_img = random_rotation(aug_img)
            aug_img = random_horizontal_flip(aug_img)

            # Save new image with "_aug" suffix
            aug_img_name = f"{os.path.splitext(img_name)[0]}_aug.jpg"
            aug_img.save(os.path.join(team_folder, aug_img_name), quality=95)

print(" Augmentation completed")
