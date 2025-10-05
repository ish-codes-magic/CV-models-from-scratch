"""Metrics calculation utilities."""

import torch


def calculate_accuracy(outputs, labels):
    """
    Calculate accuracy from model outputs and labels.
    
    Args:
        outputs (torch.Tensor): Model outputs of shape (N, num_classes)
        labels (torch.Tensor): Ground truth labels of shape (N,)
        
    Returns:
        float: Accuracy value between 0 and 1
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def get_gradient_norm(model):
    """
    Calculate the L2 norm of gradients.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        float: L2 norm of all gradients
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_weight_norm(model):
    """
    Calculate the L2 norm of model weights.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        float: L2 norm of all weights
    """
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def calculate_per_class_accuracy(predictions, labels, num_classes):
    """
    Calculate per-class accuracy.
    
    Args:
        predictions (list or numpy.ndarray): Predicted class labels
        labels (list or numpy.ndarray): Ground truth labels
        num_classes (int): Number of classes
        
    Returns:
        list: Accuracy for each class
    """
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for pred, label in zip(predictions, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    class_accuracies = [
        100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(num_classes)
    ]
    
    return class_accuracies

