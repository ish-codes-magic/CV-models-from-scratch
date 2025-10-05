"""Evaluation logic for AlexNet."""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.metrics import calculate_per_class_accuracy


class Evaluator:
    """
    Evaluator class for AlexNet model.
    """
    
    def __init__(self, model, test_loader, class_names, device):
        """
        Initialize evaluator.
        
        Args:
            model (nn.Module): AlexNet model
            test_loader (DataLoader): Test data loader
            class_names (list): List of class names
            device (torch.device): Device to evaluate on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)
    
    def evaluate(self):
        """
        Evaluate model on test set.
        
        Returns:
            dict: Dictionary containing evaluation results
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        print("Evaluating model on test set...")
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # Calculate overall accuracy
        overall_accuracy = 100 * sum(class_correct) / sum(class_total)
        
        # Calculate per-class accuracy
        class_accuracies = [
            100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(self.num_classes)
        ]
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Generate classification report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=self.class_names,
            digits=4
        )
        
        results = {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return results
    
    def print_results(self, results):
        """
        Print evaluation results.
        
        Args:
            results (dict): Evaluation results dictionary
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\nOverall Test Accuracy: {results['overall_accuracy']:.2f}%")
        
        print("\nPer-Class Accuracies:")
        print("-" * 40)
        for class_name, acc in zip(self.class_names, results['class_accuracies']):
            print(f"{class_name:12}: {acc:6.2f}%")
        
        print("\n" + "-" * 60)
        print("Classification Report:")
        print("-" * 60)
        print(results['classification_report'])
        
        print("\nConfusion Matrix:")
        print("-" * 60)
        print(results['confusion_matrix'])
        print("=" * 60)

