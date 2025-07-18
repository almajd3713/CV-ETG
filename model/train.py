import json
import os
import torch
from dataset import PytorchDataset, train_valid_test_split, train_transforms, valid_test_transforms, collate_fn
from transformers import AutoModelForObjectDetection, AutoImageProcessor, TrainingArguments, Trainer
from metrics import MAPEvaluator

with open("config.json", 'r') as f:
  config = json.load(f)
  
base_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_data_metadata = os.path.join(base_dir, "..", config["locations"]["labels"])
with open(synthetic_data_metadata, 'r') as f:
    synthetic_data = json.load(f)
    
categories = synthetic_data['categories']
id2label = {index: x['name'] for index, x in enumerate(categories, start=1)}
label2id = {v: k for k, v in id2label.items()}

train_set, valid_set, test_set = train_valid_test_split(synthetic_data)

train_dataset = PytorchDataset(train_set, transform=train_transforms)
valid_dataset = PytorchDataset(valid_set, transform=valid_test_transforms)
test_dataset = PytorchDataset(test_set, transform=valid_test_transforms)

model = AutoModelForObjectDetection.from_pretrained(
    config['model']['pretrained_weights'],
    id2label=id2label,
    label2id=label2id,
    anchor_image_size=None,
    ignore_mismatched_sizes=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir=config['training']['output_dir'],
    num_train_epochs=10,
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=MAPEvaluator(
        image_processor=AutoImageProcessor.from_pretrained(config['model']['pretrained_weights']),
        threshold=0.01,
        id2label=id2label
    ),
)

trainer.train()