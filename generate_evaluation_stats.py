import json
import random

def generate_mock_per_class_metrics():
    print("Mulai simulasi testing pada 1000 gambar untuk 5 model...")
    
    # Model architecture names
    models = ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "ConvNeXt-Tiny", "ViT-B-16"]
    
    # Base accuracies for each model (increasing trend to show model improvement)
    base_acc = {
        "ResNet-50": 0.88,
        "DenseNet-121": 0.90,
        "EfficientNet-B4": 0.92,
        "ConvNeXt-Tiny": 0.94,
        "ViT-B-16": 0.95
    }
    
    # Generate mock per-class accuracies (adding slight randomness per class)
    # MES0, MES1, MES2, MES3
    classes = ["MES0", "MES1", "MES2", "MES3"]
    
    results = {}
    
    for model in models:
        results[model] = {}
        for cls in classes:
            # Add random variance between -0.04 and +0.04
            variance = random.uniform(-0.04, 0.04)
            acc = base_acc[model] + variance
            # Cap at 0.99
            acc = min(acc, 0.99)
            results[model][cls] = round(acc, 3)
            
    # Force some specific patterns to make the weighting interesting
    # e.g., ResNet-50 is surprisingly good at MES0
    results["ResNet-50"]["MES0"] = 0.935
    # e.g., ConvNeXt-Tiny is very good at MES3
    results["ConvNeXt-Tiny"]["MES3"] = 0.971
    
    print("Simulasi selesai. Mengekspor hasil ke 'per_class_metrics.json'...")
    
    with open("per_class_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Berhasil! File per_class_metrics.json telah dibuat.")

if __name__ == "__main__":
    generate_mock_per_class_metrics()
