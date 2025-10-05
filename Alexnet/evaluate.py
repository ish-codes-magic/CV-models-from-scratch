"""Evaluation script for AlexNet."""

import argparse
import yaml
import torch
from pathlib import Path

from src import AlexNet, get_data_loaders, Evaluator, Visualizer
from src.data_loader import get_class_names
from utils.checkpoint import CheckpointManager


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup device for evaluation."""
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def main(args):
    """Main evaluation function."""
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(config)
    
    # Create data loaders
    print("\nLoading test dataset...")
    _, _, test_loader = get_data_loaders(config)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = AlexNet(num_classes=config['model']['num_classes'])
    
    # Load model weights
    checkpoint_manager = CheckpointManager(save_dir=config['checkpoint']['save_dir'])
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Try to find best model
        best_model_path = Path(config['checkpoint']['save_dir']) / 'model_best.pth'
        if best_model_path.exists():
            checkpoint_path = str(best_model_path)
        else:
            checkpoint_path = checkpoint_manager.find_latest_checkpoint()
            if not checkpoint_path:
                raise FileNotFoundError("No checkpoint found! Please provide a checkpoint path.")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
    
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError("Failed to load checkpoint!")
    
    # Get class names
    class_names = get_class_names()
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader, class_names, device)
    
    # Evaluate model
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(results)
    
    # Create visualizations
    if config['logging']['save_visualizations']:
        print("\nGenerating evaluation visualizations...")
        visualizer = Visualizer(save_dir=config['logging']['visualization_dir'])
        
        visualizer.plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            class_names,
            save_name='test_confusion_matrix.png'
        )
        
        visualizer.plot_per_class_accuracy(
            results['class_accuracies'],
            class_names,
            save_name='test_per_class_accuracy.png'
        )
        
        print("Evaluation visualizations saved!")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate AlexNet on CIFAR-10')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (if not provided, uses best or latest)')
    
    args = parser.parse_args()
    main(args)

