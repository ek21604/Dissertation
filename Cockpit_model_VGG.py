import os
import glob
import torch
from torch import optim, nn, utils
from torchvision import datasets, transforms, models
from torchvision.io import read_image
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import label_binarize
from itertools import cycle
import time
import cv2
from torchvision.models import efficientnet_b0
import seaborn as sns
from gradcam import GradCAM

# Set device for computation (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
training_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\Lab_Machine\\Images\\Training"
valid_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\Lab_Machine\\Images\\Validation"
testing_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\Lab_Machine\\Images\\Testing"

# Initialize TensorBoard writer for logging
writer = SummaryWriter()

# Define image transformations for validation/testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to VGG19 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Define training transformations with data augmentation
training_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),      # Random rotation up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random color adjustments
    transforms.transforms.GaussianBlur(kernel_size=3),     # Gaussian blur
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),  # Additional blur with 50% probability
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = datasets.ImageFolder(training_dir, transform=training_transform)
val_dataset = datasets.ImageFolder(valid_dir, transform=transform)
test_dataset = datasets.ImageFolder(testing_dir, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

class Classifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Load pretrained VGG19 model
        self.model = models.vgg19(pretrained=True)
        
        # Modify the final classifier layer for our 11 classes
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, 11)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Initialize epoch metrics
        self.training_epoch_loss = 0
        self.training_epoch_acc = 0
        self.validation_epoch_loss = 0
        self.validation_epoch_acc = 0
        self.train_batches = 0
        self.val_batches = 0

    def forward(self, x): 
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Accumulate metrics
        self.training_epoch_loss += loss.item()
        self.training_epoch_acc += acc.item()
        self.train_batches += 1
        return loss

    def on_train_epoch_end(self):
        #calculate average metrics
        avg_loss = self.training_epoch_loss / self.train_batches
        avg_acc = self.training_epoch_acc / self.train_batches
        
        # Store metrics
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)

        # Log to TensorBoard
        writer.add_scalars("Loss", {"Train": avg_loss}, self.current_epoch)
        writer.add_scalars("Accuracy", {"Train": avg_acc}, self.current_epoch)

        # Reset metrics
        self.training_epoch_loss = 0
        self.training_epoch_acc = 0
        self.train_batches = 0
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Accumulate metrics
        self.validation_epoch_loss += loss.item()
        self.validation_epoch_acc += acc.item()
        self.val_batches += 1

    def on_validation_epoch_end(self):
        #calculate average metrics
        avg_loss = self.validation_epoch_loss / self.val_batches
        avg_acc = self.validation_epoch_acc / self.val_batches
        
        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # Log to TensorBoard
        writer.add_scalars("Loss", {"Validation": avg_loss}, self.current_epoch)
        writer.add_scalars("Accuracy", {"Validation": avg_acc}, self.current_epoch)

        # Reset metrics
        self.validation_epoch_loss = 0
        self.validation_epoch_acc = 0
        self.val_batches = 0
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    
def apply_gradcam(model, image_tensor, class_idx, target_layer, label):
    """Apply Grad-CAM visualization to a single image"""
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(image_tensor.unsqueeze(0), class_idx)

    # Plot the image with Grad-CAM overlay
    plt.figure(figsize=(8, 8))
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image_np)
    plt.imshow(cam, cmap='jet', alpha=0.4)
    plt.title(f"Grad-CAM for: {label}")
    plt.axis("off")
    plt.show()

def show_gradcam_for_predictions(model, test_loader, target_layer, team_names):
    """Display Grad-CAM visualizations for both correct and incorrect predictions"""
    model.eval()
    correct_samples = []
    misclassified_samples = []

    # Collect samples from test set
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predicted_classes = logits.argmax(dim=1)

            for i in range(len(images)):
                if predicted_classes[i] == labels[i]:
                    correct_samples.append((images[i].cpu(), labels[i].item(), predicted_classes[i].item()))
                else:
                    misclassified_samples.append((images[i].cpu(), labels[i].item(), predicted_classes[i].item()))

    # Randomly select samples for visualization
    correct_samples = random.sample(correct_samples, min(10, len(correct_samples)))
    misclassified_samples = random.sample(misclassified_samples, min(10, len(misclassified_samples)))

    def plot_gradcam(samples, title):
        """Helper function to plot Grad-CAM visualizations"""
        plt.figure(figsize=(15, 6))
        for i, (image, actual_label, predicted_label) in enumerate(samples):
            plt.subplot(2, 5, i + 1)
            cam = GradCAM(model, target_layer).generate_cam(image.unsqueeze(0), predicted_label)

            # Overlay Grad-CAM on the image
            image_np = image.permute(1, 2, 0).cpu().numpy()
            plt.imshow(image_np)
            plt.imshow(cam, cmap='jet', alpha=0.4)
            plt.title(f"Pred: {team_names[predicted_label]}\nActual: {team_names[actual_label]}", fontsize=8)
            plt.axis("off")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    # Plot visualizations
    if correct_samples:
        plot_gradcam(correct_samples, "Correctly Predicted Images with Grad-CAM")
    if misclassified_samples:
        plot_gradcam(misclassified_samples, "Misclassified Images with Grad-CAM")

# Main execution
if __name__ == "__main__":
    # Load or train model
    model_path = "Models/VGG19.pth"
    classifier = Classifier().to(device)
    
    if os.path.exists(model_path):
        classifier.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        trainer = L.Trainer(max_epochs=30)
        trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)
        torch.save(classifier.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    writer.close()
    print("To view TensorBoard logs, run: tensorboard --logdir=runs/")

    # Test model
    classifier.eval()
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_probs = []
    misclassified_samples = []
    all_preds = []

    team_names = train_dataset.classes

    # Evaluate on test set
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            logits = classifier(images)
            probs = nn.functional.softmax(logits, dim=1)
            predicted_classes = logits.argmax(dim=1)
            
            # Update metrics
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted_classes.cpu().numpy())

            # Collect misclassified samples
            misclassified_indicies = (predicted_classes != labels).nonzero(as_tuple=True)[0]
            for idx in misclassified_indicies:
                misclassified_samples.append((images[idx].cpu(), labels[idx].cpu().item(), predicted_classes[idx].cpu().item()))

    # Print overall accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Overall Accuracy on Test Set: {accuracy:.2f}%")

    # Generate and display confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=team_names, yticklabels=team_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    report = classification_report(all_labels, all_preds, target_names=team_names)
    print("Classification Report:\n", report)

    # Plot ROC curve
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    plt.figure(figsize=(12, 8))
    for i, team in enumerate(zip(team_names)):  
        fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f' {team} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Display misclassified images
    images_per_page = 10
    num_pages = (len(misclassified_samples) + images_per_page - 1) // images_per_page

    def show_misclassified_images():
        """Display misclassified images in pages"""
        for page in range(num_pages):
            start_idx = page * images_per_page
            end_idx = min(start_idx + images_per_page, len(misclassified_samples))
            num_images = end_idx - start_idx

            if num_images == 0:
                print("No misclassified images to display.")
                return

            plt.figure(figsize=(12, 5))
            for i, idx in enumerate(range(start_idx, end_idx)):
                image, actual_label, predicted_label = misclassified_samples[idx]
                actual_team = team_names[actual_label]
                predicted_team = team_names[predicted_label]

                plt.subplot(2, 5, i + 1)
                plt.imshow(image.permute(1, 2, 0).numpy())
                plt.title(f"Pred: {predicted_team}\nActual: {actual_team}", fontsize=8)
                plt.axis("off")

            plt.tight_layout()
            plt.show()
            print(f"Page {page + 1} of {num_pages}")
            
            if page < num_pages - 1:
                input("Press Enter to continue to the next page...")

    # Run visualizations
    show_misclassified_images()
    show_gradcam_for_predictions(classifier.model, test_loader, classifier.model.features[-1], team_names)
