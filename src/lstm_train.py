import torch
import torch.nn as nn
from tqdm import tqdm
import time
from pathlib import Path
from src.eval_lstm import calculate_rouge

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation Loss"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def train_loop(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, vocab, idx2word):
    best_val_loss = float('inf')
    model_save_path = Path('./models/best_lstm_model.pth')
    model_save_path.parent.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        
        epoch_duration = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {epoch_duration:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    print("\nCalculating ROUGE on validation set with the best model...")
    model.load_state_dict(torch.load(model_save_path))
    rouge_scores = calculate_rouge(model, val_loader, vocab, idx2word, device)
    print(f"ROUGE Scores: {rouge_scores}")