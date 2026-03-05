import random
import shutil
from pathlib import Path

def split_data():
    print(" Splitting data into train/val folders...")
    
    # Check if we have annotated data
    annotated_dir = Path("data/annotated")
    if not annotated_dir.exists():
        print(" ERROR: data/annotated folder not found!")
        print(" Run: python src/annotation_tool.py first")
        return
    
    # Get all image files
    images = []
    raw_images_dir = Path("data/raw_images")
    
    # First, check for images that have corresponding labels
    for label_file in annotated_dir.glob("*.txt"):
        # Look for corresponding image in raw_images
        image_name = label_file.stem + ".jpg"
        image_path = raw_images_dir / image_name
        
        if image_path.exists():
            images.append({
                'image': image_path,
                'label': label_file,
                'name': image_name
            })
        else:
            # Try other extensions
            for ext in ['.jpg', '.png', '.jpeg']:
                image_path = raw_images_dir / (label_file.stem + ext)
                if image_path.exists():
                    images.append({
                        'image': image_path,
                        'label': label_file,
                        'name': label_file.stem + ext
                    })
                    break
    
    if not images:
        print("❌ ERROR: No matching image-label pairs found!")
        print(" Make sure:")
        print(" 1. You have images in data/raw_images/")
        print(" 2. You have labels in data/annotated/")
        print(" 3. File names match (image.jpg and image.txt)")
        return
    
    print(f"✅ Found {len(images)} labeled images")
    
    # Shuffle the data
    random.shuffle(images)
    
    # Split ratio (80% train, 20% val)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f" Training set: {len(train_images)} images")
    print(f" Validation set: {len(val_images)} images")
    
    # Create output directories
    train_img_dir = Path("data/train/images")
    train_label_dir = Path("data/train/labels")
    val_img_dir = Path("data/val/images")
    val_label_dir = Path("data/val/labels")
    
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files to train folders
    for item in train_images:
        # Copy image
        shutil.copy2(item['image'], train_img_dir / item['name'])
        # Copy label
        shutil.copy2(item['label'], train_label_dir / item['label'].name)
    
    # Copy files to val folders
    for item in val_images:
        # Copy image
        shutil.copy2(item['image'], val_img_dir / item['name'])
        # Copy label
        shutil.copy2(item['label'], val_label_dir / item['label'].name)
    
    print("\n📁 Folder structure created:")
    print(f" data/train/images/ → {len(list(train_img_dir.glob('*')))} images")
    print(f" data/train/labels/ → {len(list(train_label_dir.glob('*')))} labels")
    print(f" data/val/images/ → {len(list(val_img_dir.glob('*')))} images")
    print(f" data/val/labels/ → {len(list(val_label_dir.glob('*')))} labels")
    
    # Create a summary file
    summary = f"""Dataset Summary:
Total images: {len(images)}
Training set: {len(train_images)} images
Validation set: {len(val_images)} images

File extensions in raw_images:
"""
    raw_extensions = {}
    for img in raw_images_dir.glob("*"):
        if img.is_file():
            ext = img.suffix.lower()
            raw_extensions[ext] = raw_extensions.get(ext, 0) + 1
    
    for ext, count in raw_extensions.items():
        summary += f" {ext}: {count} files\n"
    
    with open("data/dataset_summary.txt", "w") as f:
        f.write(summary)
    
    print("\n✅ Data split complete! Now run:")
    print(" python src/train_detector.py")

if __name__ == "__main__":
    split_data()