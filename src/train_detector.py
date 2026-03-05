import yaml
from pathlib import Path
from ultralytics import YOLO
import os
import shutil
import numpy as np

def prepare_dataset():
    """Prepare dataset with proper formatting"""
    print(" Preparing dataset...")
    
    # Check dataset structure
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(" Data directory not found!")
        return False
    
    # Check if train and val directories exist
    train_img_dir = data_dir / "train" / "images"
    train_label_dir = data_dir / "train" / "labels"
    val_img_dir = data_dir / "val" / "images"
    val_label_dir = data_dir / "val" / "labels"
    
    # Create directories if they don't exist
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Count images and labels
    train_images = len(list(train_img_dir.glob("*")))
    train_labels = len(list(train_label_dir.glob("*")))
    val_images = len(list(val_img_dir.glob("*")))
    val_labels = len(list(val_label_dir.glob("*")))
    
    print(f"Training images: {train_images}, labels: {train_labels}")
    print(f"Validation images: {val_images}, labels: {val_labels}")
    
    if train_images == 0:
        print(" No training images found!")
        print(f" Expected in: {train_img_dir}")
        return False
    
    if train_labels == 0 and train_images > 0:
        print(" Warning: Training images found but no labels!")
        print(" Run: python scripts/split_data.py to create labels")
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images' if (data_dir / "test" / "images").exists() else 'val/images',
        'names': {0: 'walkie_talkie', 1: 'screen'}, # Use dict format
        'nc': 2
    }
    
    dataset_path = data_dir / "dataset.yaml"
    with open(dataset_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f" Dataset YAML created: {dataset_path}")
    
    # Create a simple dataset summary
    summary = f"""Dataset Summary:
Total training images: {train_images}
Total training labels: {train_labels}
Total validation images: {val_images}
Total validation labels: {val_labels}

Note: Ensure each image has a corresponding .txt label file with the same name.
"""
    
    with open(data_dir / "dataset_summary.txt", "w") as f:
        f.write(summary)
    
    return True

def train_model(config_path="configs/settings.yaml"):
    """Train YOLO model on walkie talkie data with enhanced parameters"""
    
    print(" WALKIE TALKIE DETECTOR TRAINING")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare dataset
    if not prepare_dataset():
        print(" Dataset preparation failed!")
        print("\nPlease run these steps first:")
        print("1. python scripts/capture_variations.py - to collect images")
        print("2. python src/annotation_tool.py - to label images")
        print("3. python scripts/split_data.py - to split into train/val")
        return None
    
    # Load model
    model_cfg = config.get('model')
    train_cfg = config.get('training')
    
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "Missing or invalid 'model' section in config. "
            "Expected keys like: model.name, model.input_size, model.pretrained."
        )
    
    if not isinstance(train_cfg, dict):
        raise ValueError(
            "Missing or invalid 'training' section in config. "
            "Expected keys like: training.epochs, training.batch_size, training.learning_rate."
        )
    
    if 'name' not in model_cfg:
        raise ValueError(
            "Config error: 'model.name' is missing. "
            "If your YAML has multiple top-level 'model:' blocks, the later one overwrites the earlier one. "
            "Ensure training config is under 'model:' and inference config uses a different key (e.g. 'detector:')."
        )
    
    print(f"Model: {model_cfg['name']}")
    print(f"Epochs: {train_cfg['epochs']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Device: {train_cfg['device']}")
    print(f"Learning rate: {train_cfg['learning_rate']}")
    
    # Load base model first (may be overridden by resume checkpoint)
    if model_cfg['pretrained']:
        try:
            model = YOLO(f"{model_cfg['name']}.pt")
            print(f" Loaded pretrained {model_cfg['name']}")
        except Exception as e:
            print(f" Failed to load pretrained model: {e}")
            print("Loading base YOLO model...")
            model = YOLO('yolov8n.pt')  # Load default
    else:
        model = YOLO(f"{model_cfg['name']}.yaml")
        print(f" Created new {model_cfg['name']} model")
    
    # Updated training parameters for YOLOv8 compatibility
    train_params = {
        'data': str(Path("data/dataset.yaml")),
        'epochs': train_cfg['epochs'],
        'batch': train_cfg['batch_size'],
        'imgsz': model_cfg['input_size'],
        'device': train_cfg['device'],
        'patience': train_cfg['patience'],
        'save_period': train_cfg['save_period'],
        
        # Learning rate
        'lr0': train_cfg['learning_rate'], # Initial learning rate
        
        # Basic augmentation
        'hsv_h': 0.015, # Image HSV-Hue augmentation
        'hsv_s': 0.7, # Image HSV-Saturation augmentation
        'hsv_v': 0.4, # Image HSV-Value augmentation
        'degrees': 0.0, # Image rotation
        'translate': 0.1, # Image translation
        'scale': 0.5, # Image scale
        'flipud': 0.0, # Image flip up-down
        'fliplr': 0.5, # Image flip left-right
        
        # Project settings
        'project': 'models/trained',
        'name': 'walkie_detector',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'save_period': 5,
        'cache': False,
        'single_cls': False,
        'amp': True, # Automatic Mixed Precision
        
        # Optimization
        'optimizer': 'SGD', # SGD, Adam, AdamW, NAdam, etc.
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    }
    
    # Resume logic: if a previous run exists, resume from last.pt so the new epochs value
    # (e.g. 200) is treated as the new TOTAL epochs target.
    try:
        last_candidates = []
        last_candidates.append(Path("models/trained") / "walkie_detector" / "weights" / "last.pt")
        last_candidates.append(Path("runs") / "detect" / "models" / "trained" / "walkie_detector" / "weights" / "last.pt")
        last_candidates.append(Path("runs") / "detect" / "train" / "weights" / "last.pt")
        last_pt = next((p for p in last_candidates if p.exists()), None)
        if last_pt is not None and os.getenv("WALKIE_TRAIN_FRESH", "0").strip().lower() not in ["1", "true", "yes", "y"]:
            # Ultralytics resume=True restores previous run args (including old epochs),
            # which can prevent extending epochs (e.g. 100 -> 200). So:
            # - If desired epochs > previous epochs: load last.pt but use resume=False
            # - Else: use resume=True
            prev_epochs = None
            try:
                args_candidates = [
                    last_pt.parents[1] / "args.yaml",
                    last_pt.parents[1] / "args.yml",
                ]
                args_path = next((p for p in args_candidates if p.exists()), None)
                if args_path is not None:
                    with open(args_path, "r", encoding="utf-8") as f:
                        args_yaml = yaml.safe_load(f) or {}
                    prev_epochs = args_yaml.get("epochs")
                    try:
                        prev_epochs = int(prev_epochs)
                    except Exception:
                        prev_epochs = None
            except Exception:
                prev_epochs = None

            desired_epochs = None
            try:
                desired_epochs = int(train_cfg.get("epochs"))
            except Exception:
                desired_epochs = None

            model = YOLO(str(last_pt))
            if desired_epochs is not None and prev_epochs is not None and desired_epochs > prev_epochs:
                print(f" Continuing from checkpoint (extend epochs {prev_epochs} -> {desired_epochs}): {last_pt}")
                train_params['resume'] = False
            else:
                print(f" Resuming from checkpoint: {last_pt}")
                train_params['resume'] = True
        else:
            train_params['resume'] = False
    except Exception as e:
        print(f" [WARN] Resume detection failed: {e}")
        train_params['resume'] = False
    
    # Train the model
    print("\n Starting training...")
    print("This may take a while. Please wait...")
    
    try:
        results = model.train(**train_params)
        print(f"\n Training completed!")
    except Exception as e:
        print(f"\n Training failed: {e}")
        print("\nTrying with minimal parameters...")
        
        # Try with minimal parameters
        minimal_params = {
            'data': str(Path("data/dataset.yaml")),
            'epochs': 50,
            'imgsz': 640,
            'device': train_cfg['device'],
            'project': 'models/trained',
            'name': 'walkie_detector',
            'exist_ok': True,
        }
        
        results = model.train(**minimal_params)
        print(f"\n Training completed with minimal parameters!")
    
    # Validate
    print("\n Validating model...")
    try:
        metrics = model.val()

        def _as_float(value):
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                pass
            try:
                arr = np.asarray(value)
                if arr.size == 0:
                    return None
                return float(arr.mean())
            except Exception:
                return None
        
        print(f"Validation Results:")
        map50 = _as_float(getattr(metrics.box, 'map50', None))
        map5095 = _as_float(getattr(metrics.box, 'map', None))
        precision = _as_float(getattr(metrics.box, 'p', None))
        recall = _as_float(getattr(metrics.box, 'r', None))

        if map50 is not None:
            print(f" mAP50: {map50:.4f}")
        if map5095 is not None:
            print(f" mAP50-95: {map5095:.4f}")
        if precision is not None:
            print(f" Precision: {precision:.4f}")
        if recall is not None:
            print(f" Recall: {recall:.4f}")
    except Exception as e:
        print(f" Validation failed: {e}")
    
    # Save the best model to easy location
    best_model_path = Path("models/trained/walkie_detector/weights/best.pt")
    search_roots = [Path("models/trained"), Path("runs")]

    found_best = None
    if best_model_path.exists():
        found_best = best_model_path
    else:
        for root in search_roots:
            if not root.exists():
                continue
            for possible_path in root.rglob("best.pt"):
                found_best = possible_path
                break
            if found_best is not None:
                break

    if found_best is not None:
        if found_best != best_model_path:
            print("\n Best model not found in expected location")
            print(f"Found model at: {found_best}")

        easy_access_path = Path("walkie_detector.pt")
        shutil.copy2(found_best, easy_access_path)
        print(f"\n Model saved to: {easy_access_path}")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        shutil.copy2(found_best, models_dir / "walkie_detector.pt")
        print(f" Model saved to: {models_dir / 'walkie_detector.pt'}")
    else:
        print("\n Best model not found")
    
    # Export to ONNX for faster inference (optional)
    try:
        model.export(format="onnx", imgsz=model_cfg['input_size'])
        print(f" Model exported to ONNX format")
    except Exception as e:
        print(f" ONNX export failed: {e}")
    
    print("\n Training complete! You can now run:")
    print(" python src/main.py")
    
    return model

def quick_train(epochs=30):
    """Quick training function for testing"""
    print(" Quick training for testing...")
    
    # Create a simple model
    model = YOLO('yolov8n.pt')
    
    # Quick training with minimal parameters
    results = model.train(
        data='data/dataset.yaml',
        epochs=epochs,
        imgsz=640,
        device='cpu',
        project='models/trained',
        name='quick_train',
        exist_ok=True,
        verbose=True
    )
    
    print(f" Quick training completed for {epochs} epochs")
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train walkie talkie detector")
    parser.add_argument("--quick", action="store_true", help="Run quick training (30 epochs)")
    parser.add_argument("--config", default="configs/settings.yaml", help="Configuration file")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh run (do not resume from last.pt)")
    
    args = parser.parse_args()
    
    if args.fresh:
        os.environ["WALKIE_TRAIN_FRESH"] = "1"

    if args.quick:
        quick_train()
    else:
        train_model(args.config)