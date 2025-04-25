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

# Set device for training (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data directories
training_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\Lab_Machine\\Images\\Training"
valid_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\Lab_Machine\\Images\\Validation"
testing_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\Lab_Machine\\Images\\Testing"
# cockpit:0 other:1

# Initialize TensorBoard writer for logging
writer = SummaryWriter()

# Define image transformations for validation/testing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define image transformations for training with data augmentation
training_transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.transforms.GaussianBlur(kernel_size=3),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create datasets and dataloaders
train_dataset = datasets.ImageFolder(training_dir, transform=training_transform)
val_dataset = datasets.ImageFolder(valid_dir, transform=transform)
test_dataset = datasets.ImageFolder(testing_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

class Classifier(L.LightningModule):
    """
    Lightning Module for EfficientNet-based image classification.
    This class handles the model architecture, training, validation, and testing logic.
    """
    def __init__(self, num_classes=11):
        super().__init__()
        # Initialize EfficientNet-B0
        self.model = efficientnet_b0(pretrained=True)
        
        # Modify the classifier head for our number of classes
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
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

        # Update metrics
        self.training_epoch_loss += loss.item()
        self.training_epoch_acc += acc.item()
        self.train_batches += 1
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.training_epoch_loss / self.train_batches
        avg_acc = self.training_epoch_acc / self.train_batches
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)

        # Log metrics to TensorBoard
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

        # Update metrics
        self.validation_epoch_loss += loss.item()
        self.validation_epoch_acc += acc.item()
        self.val_batches += 1

    def on_validation_epoch_end(self):
        avg_loss = self.validation_epoch_loss / self.val_batches
        avg_acc = self.validation_epoch_acc / self.val_batches
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # Log metrics to TensorBoard
        writer.add_scalars("Loss", {"Validation": avg_loss}, self.current_epoch)
        writer.add_scalars("Accuracy", {"Validation": avg_acc}, self.current_epoch)

        # Reset metrics
        self.validation_epoch_loss = 0
        self.validation_epoch_acc = 0
        self.val_batches = 0
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target_class = output[:, class_idx]
        target_class.backward()

        gradients = self.gradients
        activations = self.activations

        # Compute weights using global average pooling
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()

        # Normalize the CAM
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        return cam

def apply_gradcam(model, image_tensor, class_idx, target_layer, label):
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(image_tensor.unsqueeze(0), class_idx)

    plt.figure(figsize=(8, 8))
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image_np)
    plt.imshow(cam, cmap='jet', alpha=0.4)
    plt.title(f"Grad-CAM for: {label}")
    plt.axis("off")
    plt.show()

def show_gradcam_for_predictions(model, test_loader, target_layer, team_names):
    model.eval()
    correct_samples = []
    misclassified_samples = []

    # Collect samples from the test set
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

    # Plot correctly predicted images
    if correct_samples:
        plot_gradcam(correct_samples, "Correctly Predicted Images with Grad-CAM")

    # Plot misclassified images
    if misclassified_samples:
        plot_gradcam(misclassified_samples, "Misclassified Images with Grad-CAM")

#load or train model
model_path = "models/EfficientNet.pth"
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

#test model
classifier.eval()

correct_predictions = 0
total_predictions = 0
all_labels = []
all_probs = []
misclassified_samples = []
all_preds = []

team_names = train_dataset.classes

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        logits = classifier(images)
        probs = nn.functional.softmax(logits, dim=1)
        predicted_classes = logits.argmax(dim=1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_predictions += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(predicted_classes.cpu().numpy())

        misclassified_indicies = (predicted_classes != labels).nonzero(as_tuple=True)[0]
        for idx in misclassified_indicies:
            misclassified_samples.append((images[idx].cpu(), labels[idx].cpu().item(), predicted_classes[idx].cpu().item()))

#print overall acurracy
accuracy = (correct_predictions / total_predictions) * 100
print(f"Overall Accuracy on Test Set: {accuracy:.2f}%")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=team_names, yticklabels=team_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#classification report 
report = classification_report(all_labels, all_preds, target_names=team_names)
print("Classification Report:\n", report)

#plot ROC curve
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


images_per_page = 10
num_pages = (len(misclassified_samples) + images_per_page - 1) // images_per_page  # Total pages

def show_misclassified_images():
    for page in range(num_pages):
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, len(misclassified_samples))
        num_images = end_idx - start_idx

        if num_images == 0:
            print("No misclassified images to display.")
            return

        plt.figure(figsize=(12, 5))  # Adjust figure size for 5x2 layout

        for i, idx in enumerate(range(start_idx, end_idx)):
            image, actual_label, predicted_label = misclassified_samples[idx]
            actual_team = team_names[actual_label]  # Convert label index to team name
            predicted_team = team_names[predicted_label]

            plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns
            plt.imshow(image.permute(1, 2, 0).numpy())
            plt.title(f"Pred: {predicted_team}\nActual: {actual_team}", fontsize=8)
            plt.axis("off")

        plt.tight_layout()
        plt.show()
        print(f"Page {page + 1} of {num_pages}")
        
        # Pause before showing the next page
        if page < num_pages - 1:
            input("Press Enter to continue to the next page...")

# Run automated display
show_misclassified_images()
show_gradcam_for_predictions(classifier.model, test_loader, classifier.model.features[-1], team_names)
