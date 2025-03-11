import os
import glob
import torch
from torch import optim, nn, utils
from torchvision import datasets, transforms, models
from torchvision.io import read_image
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import label_binarize
from itertools import cycle
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

training_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\F1 Highlight Videos\\Images\\Training"
valid_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\F1 Highlight Videos\\Images\\Validation"
testing_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\F1 Highlight Videos\\Images\\Testing"
# cockpit:0 other:1

writer = SummaryWriter()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(training_dir, transform=transform)
val_dataset = datasets.ImageFolder(valid_dir, transform=transform)
test_dataset = datasets.ImageFolder(testing_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

class Classifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained = True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 11)
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

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

        self.training_epoch_loss += loss.item()
        self.training_epoch_acc += acc.item()
        self.train_batches += 1
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.training_epoch_loss / self.train_batches
        avg_acc = self.training_epoch_acc / self.train_batches
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)

        writer.add_scalar("Loss/train", avg_loss, self.current_epoch)
        writer.add_scalar("Accuracy/train", avg_acc, self.current_epoch)

        self.training_epoch_loss = 0
        self.training_epoch_acc = 0
        self.train_batches = 0
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.validation_epoch_loss += loss.item()
        self.validation_epoch_acc += acc.item()
        self.val_batches += 1

    def on_validation_epoch_end(self):
        avg_loss = self.validation_epoch_loss / self.val_batches
        avg_acc = self.validation_epoch_acc / self.val_batches
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        writer.add_scalar("Loss/val", avg_loss, self.current_epoch)
        writer.add_scalar("Accuracy/val", avg_acc, self.current_epoch)

        self.validation_epoch_loss = 0
        self.validation_epoch_acc = 0
        self.val_batches = 0
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
#load or train model
model_path = "model.pth"
classifier = Classifier().to(device)
if os.path.exists(model_path):
    classifier.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
else:
    trainer = L.Trainer(max_epochs=4)
    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(classifier.state_dict(), model_path)
    print(f"Model saved to {model_path}")

writer.close()
print("To view TensorBoard logs, run: tensorboard --logdir=runs/")

#loss and accuracy plots
'''plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(classifier.train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(classifier.train_accuracies, label='Training Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()'''

#test model
classifier.eval()

correct_predictions = 0
total_predictions = 0
all_labels = []
all_probs = []
misclassified_samples = []

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

        misclassified_indicies = (predicted_classes != labels).nonzero(as_tuple=True)[0]
        for idx in misclassified_indicies:
            misclassified_samples.append((images[idx].cpu(), labels[idx].cpu().item(), predicted_classes[idx].cpu().item()))

#print overall acurracy on test set
accuracy = (correct_predictions / total_predictions) * 100
print(f"Overall Accuracy on Test Set: {accuracy:.2f}%")

#plot ROC curve
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
#fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
#roc_auc = auc(fpr, tpr)
#plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.figure(figsize=(12, 8))
for i, team in enumerate(zip(team_names)):  # For each digit class
    fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f' {team} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

'''
#sample predictions
plt.figure(figsize=(12, 8))
#random_indices = random.sample(range(len(test_dataset)), 10)
num_samples = len(misclassified_samples)
for i in range(num_samples):
    image, actual_label, predicted_label = misclassified_samples[i]
    actual_team = team_names[actual_label]
    predicted_team = team_names[predicted_label]
    plt.subplot((num_samples // 5) + 1, 5, i + 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"Pred: {predicted_team} | Actual: {actual_team}", fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.show()
'''

images_per_page = 10
num_pages = (len(misclassified_samples) + images_per_page - 1) // images_per_page  # Total pages

def show_misclassified_images():
    """
    Automatically scrolls through misclassified images in pages.
    Each page shows 10 images (5 per row, 2 rows).
    """
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
