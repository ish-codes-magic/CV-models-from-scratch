"""Main training script for AlexNet."""

import argparse
import yaml
import torch
import json
from pathlib import Path
from datetime import datetime

from src import AlexNet, get_data_loaders, Trainer, Visualizer
from src.data_loader import get_class_names


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup device for training."""
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def save_training_report(config, metrics, model, save_dir='./logs'):
    """Save comprehensive training report."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    total_params, trainable_params = model.get_num_parameters()
    
    report = {
        'model_info': {
            'architecture': config['model']['name'],
            'dataset': config['dataset']['name'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'training_config': {
            'num_epochs': config['training']['num_epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'optimizer': config['optimizer']['type'],
            'momentum': config['optimizer']['momentum'],
            'weight_decay': config['optimizer']['weight_decay']
        },
        'final_metrics': {
            'final_train_accuracy': float(metrics['train_acc'][-1]),
            'final_val_accuracy': float(metrics['val_acc'][-1]),
            'final_train_loss': float(metrics['train_loss'][-1]),
            'final_val_loss': float(metrics['val_loss'][-1]),
            'best_val_accuracy': float(max(metrics['val_acc'])),
            'best_val_epoch': int(metrics['val_acc'].index(max(metrics['val_acc'])) + 1)
        },
        'training_metrics': {
            'epoch': metrics['epoch'],
            'train_loss': [float(x) for x in metrics['train_loss']],
            'train_acc': [float(x) for x in metrics['train_acc']],
            'val_loss': [float(x) for x in metrics['val_loss']],
            'val_acc': [float(x) for x in metrics['val_acc']],
            'learning_rate': [float(x) for x in metrics['learning_rate']],
            'epoch_time': [float(x) for x in metrics['epoch_time']]
        },
        'timestamp': datetime.now().isoformat()
    }
    
    report_path = save_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTraining report saved to: {report_path}")
    return str(report_path)


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup device
    device = setup_device(config)
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, valid_loader, test_loader = get_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(valid_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = AlexNet(num_classes=config['model']['num_classes'])
    total_params, trainable_params = model.get_num_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, valid_loader, config, device)
    
    # Train model
    metrics = trainer.train()
    
    # Create visualizations
    if config['logging']['save_visualizations']:
        print("\nGenerating visualizations...")
        visualizer = Visualizer(save_dir=config['logging']['visualization_dir'])
        
        visualizer.plot_training_curves(metrics)
        # visualizer.plot_learning_rate_schedule(metrics)
        visualizer.plot_gradient_weight_norms(metrics)
        visualizer.plot_training_time(metrics)
        visualizer.plot_overfitting_analysis(metrics)
        
        print("All visualizations saved!")
    
    # Save training report
    save_training_report(config, metrics, model, save_dir=config['logging']['log_dir'])
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model checkpoints: {config['checkpoint']['save_dir']}")
    print(f"TensorBoard logs: {config['logging']['tensorboard_dir']}")
    print(f"Visualizations: {config['logging']['visualization_dir']}")
    print(f"Training logs: {config['logging']['log_dir']}")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={config['logging']['tensorboard_dir']}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AlexNet on CIFAR-10')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    main(args)

