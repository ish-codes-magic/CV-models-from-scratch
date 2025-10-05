"""Checkpoint management utilities."""

import os
import glob
import torch
from pathlib import Path


class CheckpointManager:
    """
    Manage model checkpoints including saving and loading.
    """
    
    def __init__(self, save_dir, save_best_only=True):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir (str): Directory to save checkpoints
            save_best_only (bool): Whether to keep only the best checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_metric = float('-inf')
        
    def save_checkpoint(self, state, filename='checkpoint.pth', is_best=False):
        """
        Save model checkpoint.
        
        Args:
            state (dict): State dictionary containing model, optimizer, etc.
            filename (str): Checkpoint filename
            is_best (bool): Whether this is the best model so far
        """
        filepath = self.save_dir / filename
        torch.save(state, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        if is_best:
            best_filepath = self.save_dir / 'model_best.pth'
            torch.save(state, best_filepath)
            print(f"Best model saved: {best_filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load checkpoint from file.
        
        Args:
            filepath (str): Path to checkpoint file
            
        Returns:
            dict: Checkpoint state dictionary or None if file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Checkpoint not found: {filepath}")
            return None
        
        checkpoint = torch.load(filepath)
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint
    
    def find_latest_checkpoint(self):
        """
        Find the latest checkpoint in the save directory.
        
        Returns:
            str: Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = glob.glob(str(self.save_dir / 'checkpoint_epoch_*.pth'))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=os.path.getmtime)
        return latest
    
    def check_for_existing_weights(self, pattern='*.pth'):
        """
        Check if any model weights exist in the save directory.
        
        Args:
            pattern (str): File pattern to match
            
        Returns:
            list: List of paths to existing weight files
        """
        weight_files = glob.glob(str(self.save_dir / pattern))
        if weight_files:
            print(f"Found {len(weight_files)} existing weight file(s):")
            for wf in weight_files:
                print(f"  - {wf}")
        return weight_files
    
    def save_model_for_distribution(self, model, filename='alexnet_model.pth'):
        """
        Save model weights in a format suitable for distribution.
        
        Args:
            model (nn.Module): PyTorch model
            filename (str): Output filename
            
        Returns:
            str: Path to saved model file
        """
        filepath = self.save_dir / filename
        
        # Save only the model state dict for distribution
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
        }, filepath)
        
        print(f"Model saved for distribution: {filepath}")
        return str(filepath)
    
    def load_model_weights(self, model, filepath, strict=True):
        """
        Load model weights from file.
        
        Args:
            model (nn.Module): PyTorch model
            filepath (str): Path to weights file
            strict (bool): Whether to strictly enforce key matching
            
        Returns:
            nn.Module: Model with loaded weights
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        
        checkpoint = torch.load(filepath)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)
        
        print(f"Model weights loaded from: {filepath}")
        return model

