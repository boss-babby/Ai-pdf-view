import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from models import SeismicAutoencoder, SeismicUNet
from data_processing import get_data_loaders, prepare_training_data, SeismicDatasetV2

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cuda'):
    """
    Train the seismic model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            data = batch.to(device)
            
            # Forward pass
            if isinstance(model, SeismicAutoencoder):
                output, _ = model(data)
            else:
                output = model(data)
                
            loss = criterion(output, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch.to(device)
                
                if isinstance(model, SeismicAutoencoder):
                    output, _ = model(data)
                else:
                    output = model(data)
                    
                loss = criterion(output, data)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
    return train_losses, val_losses

def generate_seismic_image(model, input_data, device='cuda'):
    """
    Generate high-resolution seismic image from input data
    
    Args:
        model: Trained PyTorch model
        input_data: Input seismic data
        device: Device to run inference on
    
    Returns:
        numpy.ndarray: Generated seismic image
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
        if isinstance(model, SeismicAutoencoder):
            output, _ = model(input_tensor)
        else:
            output = model(input_tensor)
        
        return output.cpu().numpy().squeeze()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Set training parameters
    params = {
        'batch_size': 32,
        'patch_size': (128, 128),  # Time samples x Traces
        'stride': (64, 64),        # Stride for patch extraction
        'learning_rate': 0.001,
        'num_epochs': 100
    }
    
    print("Loading data...")
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        'data',
        batch_size=params['batch_size'],
        patch_size=params['patch_size'],
        stride=params['stride']
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = SeismicUNet(in_channels=1, out_channels=1)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=params['num_epochs'],
        learning_rate=params['learning_rate'],
        device=device
    )
    
    # Save model and training parameters
    print("Saving model and results...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'models/seismic_model.pth')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
