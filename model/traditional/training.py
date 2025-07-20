from dataset import TraditionalCNNDataset , train_transforms, valid_test_transforms, train_valid_test_split, collate_fn
from model import get_model
import os
import json
import torch
import wandb

with open("config.json", 'r') as f:
  config = json.load(f)
  
base_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_data_metadata = os.path.join("data", "labels.json")
with open(synthetic_data_metadata, 'r') as f:
    synthetic_data = json.load(f)
    
raw_image_data = os.path.join('data', config["locations"]["raw_images_metadata"])
with open(raw_image_data, 'r') as f:
  raw_image_data = json.load(f)

categories = raw_image_data['classes']
id2label = {index: x for index, x in enumerate(categories, start=1)}
label2id = {v: k for k, v in id2label.items()}

train_set, valid_set, test_set = train_valid_test_split(synthetic_data)

train_dataset = TraditionalCNNDataset(train_set, transform=train_transforms)
valid_dataset = TraditionalCNNDataset(valid_set, transform=valid_test_transforms)
test_dataset = TraditionalCNNDataset(test_set, transform=valid_test_transforms)

train_losses = []
valid_losses = []
def train_epoch(model, train_loader, class_criterion, bbox_criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        images = batch['pixel_values'].to(device)
        targets = torch.cat(batch['labels']).to(device)
        outputs, bboxes = model(images)

        optimizer.zero_grad()
        class_loss = class_criterion(outputs, targets)
        bbox_loss = bbox_criterion(bboxes, targets)
        total_loss += class_loss.item() + bbox_loss.item()
        loss = class_loss + bbox_loss
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

def validate_epoch(model, valid_loader, class_criterion, bbox_criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            images = batch['pixel_values'].to(device)
            targets = torch.cat(batch['labels']).to(device)
            outputs, bboxes = model(images)
            class_loss = class_criterion(outputs, targets)
            bbox_loss = bbox_criterion(bboxes, targets)
            total_loss += class_loss.item() + bbox_loss.item()
    return total_loss / len(valid_loader)

def train_and_eval(epochs, device, run=None):
    model = get_model(num_classes=len(categories)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    class_criterion = torch.nn.CrossEntropyLoss()
    bbox_criterion = torch.nn.MSELoss()  

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, class_criterion, bbox_criterion, optimizer, device)
        valid_loss = validate_epoch(model, valid_loader, class_criterion, bbox_criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        if run:
            run.log({"train_loss": train_loss, "valid_loss": valid_loss})

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(categories)).to(device)
    run = wandb.init(project="traditional_cnn_training", )
    run.watch(model)
    
    trained_model = train_and_eval(30, device)

    # Save the trained model
    model_save_path = os.path.join(base_dir, config['training']['output_dir'], 'traditional_cnn_model.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")