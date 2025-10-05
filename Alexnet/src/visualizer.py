"""Visualization utilities for training metrics and results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """
    Handle all visualization tasks for training and evaluation.
    """
    
    def __init__(self, save_dir='./visualizations'):
        """
        Initialize visualizer.
        
        Args:
            save_dir (str): Directory to save visualization plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        
    def plot_training_curves(self, metrics, save_name='training_curves.png'):
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            metrics (dict): Dictionary containing training metrics
            save_name (str): Filename for saved plot
        """
        epochs = metrics['epoch']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(epochs, metrics['train_loss'], 'r-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, metrics['val_loss'], 'b-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, metrics['train_acc'], 'g-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, metrics['val_acc'], 'orange', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved: {save_path}")
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_name='confusion_matrix.png'):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list): List of class names
            save_name (str): Filename for saved plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {save_path}")
        
    def plot_per_class_accuracy(self, class_accuracies, class_names, save_name='per_class_accuracy.png'):
        """
        Plot per-class accuracy bar chart.
        
        Args:
            class_accuracies (list): Accuracy for each class
            class_names (list): List of class names
            save_name (str): Filename for saved plot
        """
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, class_accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class accuracy plot saved: {save_path}")
        
    def plot_learning_rate_schedule(self, metrics, save_name='learning_rate_schedule.png'):
        """
        Plot learning rate schedule over epochs.
        
        Args:
            metrics (dict): Dictionary containing training metrics
            save_name (str): Filename for saved plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], metrics['learning_rate'], 'b-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning rate schedule saved: {save_path}")
        
    def plot_gradient_weight_norms(self, metrics, save_name='gradient_weight_norms.png'):
        """
        Plot gradient and weight norms over epochs.
        
        Args:
            metrics (dict): Dictionary containing training metrics
            save_name (str): Filename for saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gradient norm
        ax1.plot(metrics['epoch'], metrics['gradient_norm'], 'r-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Gradient Norm', fontsize=12)
        ax1.set_title('Gradient Norm Over Training', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Weight norm
        ax2.plot(metrics['epoch'], metrics['weight_norm'], 'b-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Weight Norm', fontsize=12)
        ax2.set_title('Weight Norm Over Training', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gradient/weight norms plot saved: {save_path}")
        
    def plot_training_time(self, metrics, save_name='training_time.png'):
        """
        Plot training time per epoch.
        
        Args:
            metrics (dict): Dictionary containing training metrics
            save_name (str): Filename for saved plot
        """
        plt.figure(figsize=(10, 6))
        plt.bar(metrics['epoch'], metrics['epoch_time'], color='lightblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training time plot saved: {save_path}")
        
    def create_comprehensive_dashboard(self, metrics, class_accuracies, class_names, 
                                      y_true, y_pred, save_name='comprehensive_dashboard.png'):
        """
        Create a comprehensive dashboard with all key metrics.
        
        Args:
            metrics (dict): Dictionary containing training metrics
            class_accuracies (list): Accuracy for each class
            class_names (list): List of class names
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            save_name (str): Filename for saved plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(metrics['epoch'], metrics['train_loss'], 'r-', label='Train', linewidth=2)
        ax1.plot(metrics['epoch'], metrics['val_loss'], 'b-', label='Val', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(metrics['epoch'], metrics['train_acc'], 'g-', label='Train', linewidth=2)
        ax2.plot(metrics['epoch'], metrics['val_acc'], 'orange', label='Val', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Training time
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(metrics['epoch'], metrics['epoch_time'], color='lightblue', alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Training Time per Epoch', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Confusion matrix
        ax4 = fig.add_subplot(gs[1, :2])
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=class_names, yticklabels=class_names)
        ax4.set_xlabel('Predicted Label')
        ax4.set_ylabel('True Label')
        ax4.set_title('Confusion Matrix', fontweight='bold')
        
        # 5. Per-class accuracy
        ax5 = fig.add_subplot(gs[1, 2])
        bars = ax5.barh(class_names, class_accuracies, color='skyblue', alpha=0.7)
        ax5.set_xlabel('Accuracy (%)')
        ax5.set_title('Per-Class Accuracy', fontweight='bold')
        ax5.set_xlim(0, 100)
        ax5.grid(axis='x', alpha=0.3)
        
        # 6. Gradient norm
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.plot(metrics['epoch'], metrics['gradient_norm'], 'r-', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Gradient Norm')
        ax6.set_title('Gradient Norm', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Weight norm
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(metrics['epoch'], metrics['weight_norm'], 'b-', linewidth=2)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Weight Norm')
        ax7.set_title('Weight Norm', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary statistics
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        summary_text = f"""
        TRAINING SUMMARY
        ─────────────────────────
        Final Train Acc: {metrics['train_acc'][-1]*100:.2f}%
        Final Val Acc: {metrics['val_acc'][-1]*100:.2f}%
        Best Val Acc: {max(metrics['val_acc'])*100:.2f}%
        
        Final Train Loss: {metrics['train_loss'][-1]:.4f}
        Final Val Loss: {metrics['val_loss'][-1]:.4f}
        
        Total Epochs: {len(metrics['epoch'])}
        Avg Time/Epoch: {np.mean(metrics['epoch_time']):.2f}s
        Total Time: {sum(metrics['epoch_time'])/60:.1f}m
        """
        ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comprehensive dashboard saved: {save_path}")
        
    def plot_overfitting_analysis(self, metrics, save_name='overfitting_analysis.png'):
        """
        Plot overfitting analysis (train-val gap).
        
        Args:
            metrics (dict): Dictionary containing training metrics
            save_name (str): Filename for saved plot
        """
        train_val_gap = [t - v for t, v in zip(metrics['train_acc'], metrics['val_acc'])]
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epoch'], train_val_gap, 'r-', linewidth=2)
        plt.fill_between(metrics['epoch'], 0, train_val_gap, alpha=0.3, color='red')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy Gap (Train - Val)', fontsize=12)
        plt.title('Overfitting Analysis', fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overfitting analysis plot saved: {save_path}")

