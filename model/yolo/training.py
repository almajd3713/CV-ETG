import os
import torch
from model import model

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml = os.path.join(base_dir, 'dataset', 'data.yaml')

if __name__ == "__main__":
    # Example usage of the YOLO model    
    model.to(device)
    model.train(data = data_yaml, project = "model/yolo/runs", epochs=50, imgsz=1000, batch=16, device=device)
    print("Training complete. Evaluating model...")
    model.eval(project = "model/yolo/runs", imgsz=1000, batch=16, device=device)
    print("Exporting model to ONNX format...")
    model.export(format='onnx', imgsz=1000, device=device)