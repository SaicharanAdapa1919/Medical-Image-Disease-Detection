import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import shutil

# Configuration
CLASSES = [
    "normal", "pneumonia", "covid19", "tuberculosis", "lung_cancer", "pneumothorax",
    "pleural_effusion", "heart_failure", "bone_fracture", "osteoporosis", "brain_tumor",
    "stroke", "alzheimers", "osteoarthritis", "liver_cirrhosis", "kidney_stones",
    "appendicitis", "breast_cancer", "diabetic_retinopathy", "melanoma"
]
IMG_SIZE = (224, 224)
TRAIN_COUNT = 100  # per class
TEST_COUNT = 20    # per class
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def generate_base_image(size):
    """Base chest/body X-ray like image"""
    img = Image.new('L', size, color=180)
    draw = ImageDraw.Draw(img)
    
    # Gradient background
    for y in range(size[1]):
        for x in range(size[0]):
            base = 170 + int(20 * np.sin(x / 30) * np.cos(y / 30))
            img.putpixel((x, y), base)
    
    # Basic anatomy (ribs, spine)
    for i in range(5):
        y_offset = 40 + i * 35
        draw.arc([20, y_offset, 110, y_offset + 40], start=180, end=360, fill=140, width=2)
        draw.arc([114, y_offset, 204, y_offset + 40], start=180, end=360, fill=140, width=2)
    
    # Heart/diaphragm
    draw.ellipse([85, 90, 139, 170], fill=120)
    draw.arc([10, 140, 214, 240], start=0, end=180, fill=150, width=3)
    
    # Noise and blur
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img.convert('RGB')

def generate_normal_image(size):
    return generate_base_image(size)

def generate_pneumonia_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Consolidations
    for _ in range(random.randint(3, 6)):
        x = random.randint(30, 90) if random.random() > 0.5 else random.randint(130, 190)
        y = random.randint(100, 180)
        r = random.randint(15, 40)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=random.randint(160, 220))
    return img

def generate_covid_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Ground-glass
    for _ in range(random.randint(4, 8)):
        x = random.randint(20, 70) if random.random() > 0.5 else random.randint(150, 200)
        y = random.randint(60, 180)
        r = random.randint(20, 50)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=random.randint(130, 180))
    return img

def generate_tuberculosis_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Cavitary lesions
    for _ in range(2, 4):
        x, y = random.randint(50, 170), random.randint(80, 160)
        draw.ellipse([x-15, y-15, x+15, y+15], fill=60)  # Cavity
        draw.ellipse([x-25, y-25, x+25, y+25], fill=200)  # Surrounding
    return img

def generate_lung_cancer_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Nodules/masses
    for _ in range(1, 3):
        x, y = random.randint(40, 180), random.randint(70, 150)
        r = random.randint(20, 35)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=240)
    return img

def generate_pneumothorax_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Pleural line
    draw.line([10, 50, 60, 50], fill=255, width=3)
    # Collapsed lung
    draw.rectangle([70, 40, 220, 220], fill=100)
    return img

def generate_pleural_effusion_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Fluid in pleural space
    draw.rectangle([20, 120, 80, 210], fill=160)
    draw.rectangle([140, 130, 200, 210], fill=160)
    return img

def generate_heart_failure_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Cardiomegaly
    draw.ellipse([60, 70, 164, 200], fill=110)
    # Pulmonary edema
    for _ in range(8):
        x = random.randint(30, 190)
        y = random.randint(50, 120)
        draw.ellipse([x-8, y-8, x+8, y+8], fill=170)
    return img

def generate_bone_fracture_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Fracture lines
    points = [(100, 100), (120, 110), (130, 105), (140, 120), (150, 115)]
    draw.line(points, fill=255, width=4)
    return img

def generate_osteoporosis_image(size):
    img = generate_base_image(size)
    arr = np.array(img)
    arr = np.clip(arr * 0.7 + np.random.normal(0, 15, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.convert('RGB')

def generate_brain_tumor_image(size):
    img = Image.new('L', size, color=160)
    draw = ImageDraw.Draw(img)
    # Brain mass
    draw.ellipse([70, 60, 150, 140], fill=220)
    # Edema
    draw.ellipse([50, 40, 170, 160], fill=140)
    arr = np.array(img)
    noise = np.random.normal(0, 10, arr.shape)
    img = Image.fromarray(np.clip(arr.astype(float) + noise, 0, 255).astype(np.uint8))
    return img.convert('RGB')

def generate_stroke_image(size):
    img = generate_brain_tumor_image(size)  # Reuse base
    draw = ImageDraw.Draw(img)
    # Hypodense wedge
    draw.polygon([(80, 90), (120, 100), (100, 140)], fill=80)
    return img

def generate_alzheimers_image(size):
    img = generate_brain_tumor_image(size)
    draw = ImageDraw.Draw(img)
    # Ventricular enlargement
    draw.ellipse([95, 85, 125, 115], fill=50)
    draw.ellipse([100, 130, 120, 150], fill=50)
    return img

def generate_osteoarthritis_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Bone spurs at joints
    draw.polygon([(180, 180), (195, 170), (205, 185)], fill=255)
    return img

def generate_liver_cirrhosis_image(size):
    img = Image.new('L', size, color=170)
    draw = ImageDraw.Draw(img)
    # Nodular liver
    for _ in range(10):
        x = random.randint(100, 160)
        y = random.randint(120, 190)
        r = random.randint(5, 12)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=random.randint(140, 200))
    return img.convert('RGB')

def generate_kidney_stones_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # Calcifications
    for _ in range(3, 6):
        x = random.randint(110, 150)
        y = random.randint(140, 180)
        draw.ellipse([x-3, y-3, x+3, y+3], fill=255)
    return img

def generate_appendicitis_image(size):
    img = generate_base_image(size)
    draw = ImageDraw.Draw(img)
    # RLQ thickening
    draw.ellipse([160, 170, 200, 210], fill=200)
    return img

def generate_breast_cancer_image(size):
    img = Image.new('L', size, color=190)
    draw = ImageDraw.Draw(img)
    # Spiculated mass
    draw.ellipse([140, 60, 180, 100], fill=240)
    # Spicules
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        dx, dy = 20 * np.cos(rad), 20 * np.sin(rad)
        draw.line([160, 80, 160+dx, 80+dy], fill=240, width=2)
    return img.convert('RGB')

def generate_diabetic_retinopathy_image(size):
    img = Image.new('L', size, color=120)
    draw = ImageDraw.Draw(img)
    # Microaneurysms/hemorrhages
    for _ in range(20):
        x, y = random.randint(40, 180), random.randint(40, 180)
        draw.ellipse([x-1, y-1, x+1, y+1], fill=20)
    # Vessels
    draw.line([110, 110, 130, 130], fill=50, width=1)
    return img.convert('RGB')

def generate_melanoma_image(size):
    img = Image.new('L', size, color=200)
    draw = ImageDraw.Draw(img)
    # Irregular lesion
    points = [(100,100), (120,90), (140,110), (130,130), (110,125)]
    draw.polygon(points, fill=40)
    draw.ellipse([105, 102, 115, 112], fill=20)  # Variation
    return img.convert('RGB')

# Generator dictionary
GENERATORS = {
    "normal": generate_normal_image,
    "pneumonia": generate_pneumonia_image,
    "covid19": generate_covid_image,
    "tuberculosis": generate_tuberculosis_image,
    "lung_cancer": generate_lung_cancer_image,
    "pneumothorax": generate_pneumothorax_image,
    "pleural_effusion": generate_pleural_effusion_image,
    "heart_failure": generate_heart_failure_image,
    "bone_fracture": generate_bone_fracture_image,
    "osteoporosis": generate_osteoporosis_image,
    "brain_tumor": generate_brain_tumor_image,
    "stroke": generate_stroke_image,
    "alzheimers": generate_alzheimers_image,
    "osteoarthritis": generate_osteoarthritis_image,
    "liver_cirrhosis": generate_liver_cirrhosis_image,
    "kidney_stones": generate_kidney_stones_image,
    "appendicitis": generate_appendicitis_image,
    "breast_cancer": generate_breast_cancer_image,
    "diabetic_retinopathy": generate_diabetic_retinopathy_image,
    "melanoma": generate_melanoma_image
}

def create_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Clear existing dataset
    dataset_dir = os.path.join(base_dir, "dataset")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        print("Cleared existing dataset directory.")
    
    for split in ["train", "test"]:
        for cls in CLASSES:
            dir_path = os.path.join(base_dir, "dataset", split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            count = TRAIN_COUNT if split == "train" else TEST_COUNT
            
            for i in range(count):
                img = GENERATORS[cls](IMG_SIZE)
                filename = f"{cls}_{i:03d}.png"
                img.save(os.path.join(dir_path, filename))
            
            print(f"Created {count} {cls} images in dataset/{split}/{cls}")

if __name__ == "__main__":
    print("Generating expanded synthetic medical imaging dataset with 20 diseases...")
    print(f"Classes: {CLASSES}")
    create_dataset()
    print("Done! Dataset ready for training with maximum feasible human diseases represented.")

