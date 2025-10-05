"""Training logic for AlexNet."""

import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from utils.metrics import calculate_accuracy, get_gradient_norm, get_weight_norm
from utils.checkpoint import CheckpointManager


class Trainer:
    """
    Trainer class for AlexNet model.
    """
    
    def __init__(self, model, train_loader, valid_loader, config, device):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): AlexNet model
            train_loader (DataLoader): Training data loader
            valid_loader (DataLoader): Validation data loader
            config (dict): Configuration dictionary
            device (torch.device): Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.device = device
        
        # Training parameters
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        
        # Setup criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=config['checkpoint']['save_dir'],
            save_best_only=config['checkpoint']['save_best_only']
        )
        
        # Check for existing weights
        # existing_weights = self.checkpoint_manager.check_for_existing_weights()
        
        # Setup TensorBoard
        log_dir = Path(config['logging']['tensorboard_dir']) / f"alexnet_cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.log_dir = log_dir
        
        # Metrics storage
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': [],
            'gradient_norm': [],
            'weight_norm': []
        }
        
        self.best_val_acc = 0.0
        self.start_epoch = 0
        
        # Resume from checkpoint if specified
        resume_path = config['checkpoint'].get('resume_from')
        if resume_path:
            self._resume_from_checkpoint(resume_path)
    
    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=opt_config['momentum'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
        
        return optimizer
    
    def _resume_from_checkpoint(self, checkpoint_path):
        """Resume training from checkpoint."""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        if checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.metrics = checkpoint.get('metrics', self.metrics)
            print(f"Resumed from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            tuple: (avg_loss, avg_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        total_steps = len(self.train_loader)
        log_frequency = self.config['logging']['log_frequency']
        
        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Calculate gradient norm before optimizer step
            grad_norm = get_gradient_norm(self.model)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            running_corrects += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Log to TensorBoard
            if (i + 1) % log_frequency == 0:
                global_step = epoch * total_steps + i
                self.writer.add_scalar('Training/Loss_Step', loss.item(), global_step)
                self.writer.add_scalar('Training/Accuracy_Step',
                                     calculate_accuracy(outputs, labels), global_step)
                self.writer.add_scalar('Training/Gradient_Norm', grad_norm, global_step)
                self.writer.add_scalar('Training/Weight_Norm',
                                     get_weight_norm(self.model), global_step)
                
                print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                      f'Step [{i+1}/{total_steps}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {calculate_accuracy(outputs, labels):.4f}')
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc, grad_norm
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            tuple: (avg_loss, avg_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in self.valid_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                running_corrects += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop."""
        print("=" * 50)
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"TensorBoard logs: {self.log_dir}")
        print("=" * 50)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc, grad_norm = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.metrics['epoch'].append(epoch + 1)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['learning_rate'].append(current_lr)
            self.metrics['epoch_time'].append(epoch_time)
            self.metrics['gradient_norm'].append(grad_norm)
            self.metrics['weight_norm'].append(get_weight_norm(self.model))
            
            # Log to TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Epoch/Time', epoch_time, epoch)
            
            # Print epoch summary
            print(f'\nEpoch [{epoch+1}/{self.num_epochs}] Summary:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            print('-' * 50)
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            save_frequency = self.config['checkpoint']['save_frequency']
            if (epoch + 1) % save_frequency == 0 or is_best:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'metrics': self.metrics,
                    'config': self.config
                }
                
                filename = f'checkpoint_epoch_{epoch+1}.pth'
                self.checkpoint_manager.save_checkpoint(checkpoint, filename, is_best)
        
        # Save final model for distribution
        self.checkpoint_manager.save_model_for_distribution(
            self.model,
            filename='alexnet_cifar10_final.pth'
        )
        
        # Close TensorBoard writer
        self.writer.close()
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"TensorBoard logs saved to: {self.log_dir}")
        print("=" * 50)
        
        return self.metrics

