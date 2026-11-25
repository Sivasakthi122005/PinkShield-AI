import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import json
import os

def train_model(model, train_loader, val_loader, device,
                epochs=25, save_path='models/efficientnet_best.pt'):

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("Starting training...\n")

    for epoch in range(epochs):
        # ---- TRAINING PHASE ----
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        # ---- VALIDATION PHASE ----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        # ---- METRICS ----
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}")

        scheduler.step()

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")
    _save_final_metrics(model, val_loader, device, save_path,
                        getattr(val_loader, 'split_description', 'n/a'))

    # Save training history for dashboard charts
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)
    print("Training history saved to models/training_history.json")

    return history


def _save_final_metrics(model, val_loader, device, save_path, split_desc):
    """Reload the best checkpoint and store sensitivity / specificity / AUC."""
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, y in val_loader:
            p = model(images.to(device)).softmax(dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist()); labels.extend(y.numpy().tolist())
    probs, labels = np.array(probs), np.array(labels)
    tn, fp, fn, tp = confusion_matrix(labels, probs >= 0.5, labels=[0, 1]).ravel()
    metrics = {
        'accuracy': round(100 * (tp + tn) / len(labels), 2),
        'sensitivity': round(100 * tp / max(tp + fn, 1), 2),
        'specificity': round(100 * tn / max(tn + fp, 1), 2),
        'auc_roc': round(float(roc_auc_score(labels, probs)), 4),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'split': split_desc,
    }
    with open('models/image_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Image metrics saved:", metrics)
