import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import os
import wandb
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy

# Create a directory for saving checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

wandb_username = "aslane84"  
project_name = "turkish_product_reviews"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the dataset
dataset = load_dataset("turkish_product_reviews")

# Load a tokenizer and model for Turkish
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-128k-uncased", num_labels=2)

# Split the training dataset into training and validation sets (80% train, 20% validation)
train_dataset, validation_dataset = dataset["train"].train_test_split(test_size=0.2).values()

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization to both training and validation datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch
tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'sentiment'])
tokenized_validation_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'sentiment'])

# Create dataloaders
batch_size=64
lr = 1e-3

# Create DataLoaders
train_loader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(tokenized_validation_dataset, batch_size=batch_size)

# for batch in train_loader:
#     print(batch)
#     exit()# Just to print the first batch and not iterate over the entire dataset
# exit()


# Check if GPUs are available and set the model to use them
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    # If you have multiple GPUs, you can specify them by index, e.g., [0, 1]
    model = DataParallel(model, device_ids=[0])
else:
    device = torch.device("cpu")
    model.to(device)


# Define optimizer and loss function
# optimizer = AdamW(model.parameters(), lr=5e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_function = nn.BCEWithLogitsLoss()

# Number of training epochs
epochs = 4

# For updating learning rate
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)

# Login to wandb
wandb.login()

# Initialize a new wandb run
wandb.init(project=project_name, entity=wandb_username)

# Configurations for hyperparameter tracking
wandb.config = {
  "learning_rate": lr,
  "epochs": epochs,
  "batch_size": batch_size
}

print("******************Training started******************")

# Training loop
for epoch in range(epochs):
    model.train()
    train_preds, train_labels, total_train_loss = [], [], 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['sentiment'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_train_loss += loss.item()
        train_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_precision = precision_score(train_labels, train_preds, average='binary')
    train_recall = recall_score(train_labels, train_preds, average='binary')
    train_f1 = f1_score(train_labels, train_preds, average='binary')

    wandb.log({"epoch": epoch, 
               "train_loss": avg_train_loss,
               "train_accuracy": train_accuracy, 
               "train_precision": train_precision,
               "train_recall": train_recall,
               "train_f1": train_f1}, commit=False)

        
    # Save a checkpoint at the end of each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_train_loss
    }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
    
    average_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch: {epoch}, Loss: {average_train_loss}")

    # Validation loop
    model.eval()
    val_preds, val_labels = [], []
    total_val_loss = 0
    
    for batch in tqdm(valid_loader, desc="Validating", leave=False, disable=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['sentiment'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_val_loss += outputs.loss.item()
        val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(valid_loader)
    val_precision = precision_score(val_labels, val_preds, average='binary')
    val_recall = recall_score(val_labels, val_preds, average='binary')
    val_f1 = f1_score(val_labels, val_preds, average='binary')

    wandb.log({"epoch": epoch, 
               "val_loss": avg_val_loss,
               "val_precision": val_precision, 
               "val_recall": val_recall,
               "val_f1": val_f1}, commit=True)
        
wandb.finish()