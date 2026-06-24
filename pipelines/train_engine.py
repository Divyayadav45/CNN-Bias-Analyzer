# pipelines/train_engine.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def run_training_pipeline(model, train_loader, val_loader, model_name="resnet18", epochs=10, lr=0.001):
    """Unified training loop with checkpoint orchestration and TensorBoard serialization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter(log_dir=f"./logs/{model_name}")
    
    best_acc = 0.0
    os.makedirs("./checkpoints", exist_ok=True)
    
    print(f"🚀 Initializing Training Pipeline for {model_name} on target hardware device: {device}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        # Validation evaluation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_epoch_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_acc, epoch)
        writer.add_scalar("Loss/Validation", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save optimal checkpoint weights matrix
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./checkpoints/{model_name}_best.pth")
            
    writer.close()
    print(f"✨ Training pipeline finalized. Best validation accuracy: {best_acc:.2f}%")