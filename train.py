import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from model import MultiTaskModel
os.makedirs("saved_model", exist_ok=True)
df = pd.read_csv("data/clean_data.csv")
df = df.dropna(subset=["tweet"])
df["tweet"] = df["tweet"].astype(str)
df["rationale"] = df["rationale"].fillna("").astype(str)
# ---------------- TARGET ENCODING & WEIGHTS ----------------
le = LabelEncoder()
df["Target"] = le.fit_transform(df["Target"])
np.save("saved_model/target_classes.npy", le.classes_)
# Bully weights
labels = df["Bully_Label"].values
all_classes = np.array([0, 1])
b_weights = compute_class_weight(class_weight="balanced", classes=all_classes, y=labels)
class_weights = torch.tensor(b_weights, dtype=torch.float)
# Target weights
target_labels_array = df["Target"].values
target_classes_array = np.unique(target_labels_array)
t_weights = compute_class_weight(class_weight="balanced", classes=target_classes_array, y=target_labels_array)
target_class_weights = torch.tensor(t_weights, dtype=torch.float)
# ---------------- TOKENIZER & DATASET ----------------
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/hing-bert")
class BullyDataset(Dataset):
    def __init__(self, texts, bullies, targets, rationales):
        self.texts = texts.tolist()
        self.bully = bullies.tolist()
        self.target = targets.tolist()
        self.rationale = rationales.tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        rationale_string = self.rationale[idx].lower()
        encoding = tokenizer(
            text, padding="max_length", truncation=True, 
            max_length=128, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        rationale_labels = []
        for token in tokens:
            clean_token = token.replace("##", "").lower() 
            if clean_token in ["[CLS]", "[SEP]", "[PAD]"] or len(clean_token) < 2:
                rationale_labels.append(0)
            elif clean_token in rationale_string:
                rationale_labels.append(1)
            else:
                rationale_labels.append(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bully": torch.tensor(self.bully[idx], dtype=torch.long),
            "target": torch.tensor(self.target[idx], dtype=torch.long),
            "rationale": torch.tensor(rationale_labels, dtype=torch.long)
        }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rationale_weights = torch.tensor([1.0, 10.0]).to(device)
# ---------------- K-FOLD SETUP ----------------
k_folds = 5
epochs = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []
absolute_best_val_loss = float('inf')
print(f"Starting {k_folds}-Fold Cross Validation ({epochs} Epochs per fold)")
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["Bully_Label"])):
    print(f"\n========================================")
    print(f"             FOLD {fold + 1}/{k_folds}             ")
    print(f"========================================")
    # 1. Split Data for this fold
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    train_dataset = BullyDataset(train_data["tweet"], train_data["Bully_Label"], train_data["Target"], train_data["rationale"])
    val_dataset = BullyDataset(val_data["tweet"], val_data["Bully_Label"], val_data["Target"], val_data["rationale"])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    # 2. Reset Model & Optimizer for fresh learning
    model = MultiTaskModel(num_targets=len(le.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)   
    bully_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    target_criterion = nn.CrossEntropyLoss(weight=target_class_weights.to(device))
    rationale_criterion = nn.CrossEntropyLoss(weight=rationale_weights)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    best_fold_val_loss = float('inf')
    best_fold_metrics = {}
    # 3. Train the Fold
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bully_labels = batch["bully"].to(device)
            target_labels = batch["target"].to(device)
            rationale_labels = batch["rationale"].to(device)
            optimizer.zero_grad()
            bully_logits, target_logits, rationale_logits = model(input_ids, attention_mask)
            loss1 = bully_criterion(bully_logits, bully_labels)
            loss2 = target_criterion(target_logits, target_labels)          
            active_loss = attention_mask.view(-1) == 1
            active_logits = rationale_logits.view(-1, 2)[active_loss]
            active_labels = rationale_labels.view(-1)[active_loss]
            loss3 = rationale_criterion(active_logits, active_labels)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            scheduler.step() 
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        # 4. Evaluate the Fold
        model.eval()
        val_loss = 0
        bully_preds, bully_true, target_preds, target_true, rat_preds, rat_true = [], [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bully_labels = batch["bully"].to(device)
                target_labels = batch["target"].to(device)
                rationale_labels = batch["rationale"].to(device)
                bully_logits, target_logits, rationale_logits = model(input_ids, attention_mask)
                loss1 = bully_criterion(bully_logits, bully_labels)
                loss2 = target_criterion(target_logits, target_labels)                
                active_loss = attention_mask.view(-1) == 1
                active_logits = rationale_logits.view(-1, 2)[active_loss]
                active_labels = rationale_labels.view(-1)[active_loss]
                loss3 = rationale_criterion(active_logits, active_labels)               
                val_loss += (loss1 + loss2 + loss3).item()
                bully_predictions = torch.argmax(bully_logits, dim=1)
                target_predictions = torch.argmax(target_logits, dim=1)
                rationale_predictions = torch.argmax(rationale_logits, dim=2) 
                bully_preds.extend(bully_predictions.cpu().numpy())
                bully_true.extend(bully_labels.cpu().numpy())
                target_preds.extend(target_predictions.cpu().numpy())
                target_true.extend(target_labels.cpu().numpy())
                rat_preds.extend(rationale_predictions.view(-1)[active_loss].cpu().numpy())
                rat_true.extend(active_labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        bully_acc = accuracy_score(bully_true, bully_preds)
        target_acc = accuracy_score(target_true, target_preds)
        rat_acc = accuracy_score(rat_true, rat_preds) 
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Bully Acc: {bully_acc:.4f} | Target Acc: {target_acc:.4f}")
        # Track best epoch in this fold
        if avg_val_loss < best_fold_val_loss:
            best_fold_val_loss = avg_val_loss
            best_fold_metrics = {"Bully": bully_acc, "Target": target_acc, "Rationale": rat_acc}           
            # Save if it is the absolute best model across ALL folds
            if avg_val_loss < absolute_best_val_loss:
                absolute_best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "saved_model/model.pt")
                print("New Global Best Model Saved!")
    fold_results.append(best_fold_metrics)
    print(f"Fold {fold+1} Best -> Bully: {best_fold_metrics['Bully']:.4f} | Target: {best_fold_metrics['Target']:.4f}")
# ---------------- FINAL AVERAGES ----------------
print("\n" + "="*40)
print("FINAL K-FOLD CROSS VALIDATION RESULTS")
print("="*40)
avg_bully = np.mean([f['Bully'] for f in fold_results])
avg_target = np.mean([f['Target'] for f in fold_results])
avg_rat = np.mean([f['Rationale'] for f in fold_results])
print(f"Average Bully Accuracy:     {avg_bully:.4f}")
print(f"Average Target Accuracy:    {avg_target:.4f}")
print(f"Average Rationale Accuracy: {avg_rat:.4f}")
print("="*40)
print("The absolute best model from all folds is saved and ready for predict.py!")