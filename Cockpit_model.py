import os
import glob
import torch
from torch import optim, nn, utils
from torchvision import datasets, transforms
from torchvision.io import read_image
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

training_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\F1 Highlight Videos\\Images\\Training"
valid_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\F1 Highlight Videos\\Images\\Validation"
testing_dir = "C:\\Users\\price\\Documents\\Uni\\Dis\\F1 Highlight Videos\\Images\\Testing"
# cockpit:0 other:1

writer = SummaryWriter("runs/F1_Classifier")

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
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.train_losses = []
        self.train_accuracies = []

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.train_losses.append(loss.item())
        self.train_accuracies.append(acc.item())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        writer.add_scalar("Loss/train", loss, batch_idx)
        writer.add_scalar("Accuracy/train", acc, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        writer.add_scalar("Loss/val", loss, batch_idx)
        writer.add_scalar("Accuracy/val", acc, batch_idx)
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
#load or train model
model_path = "f1_model.pth"
classifier = Classifier().to(device)
if os.path.exists(model_path):
    classifier.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
else:
    trainer = L.Trainer(max_epochs=2)
    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(classifier.state_dict(), model_path)
    print(f"Model saved to {model_path}")

writer.close()
print("To view TensorBoard logs, run: tensorboard --logdir=runs/F1_Classifier")

#loss and accuracy plots
plt.figure(figsize=(12, 5))

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
plt.show()

#test model
classifier.eval()
correct_predictions = 0
total_predictions = 0
all_labels = []
all_probs = []
misclassified_samples = []

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
fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#sample predictions
plt.figure(figsize=(12, 6))
#random_indices = random.sample(range(len(test_dataset)), 10)
num_samples = min(10, len(misclassified_samples))
for i in range(num_samples):
    image, actual_label, predicted_label = misclassified_samples[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"Pred: {predicted_label} | Actual: {actual_label}")
    plt.axis("off")
plt.tight_layout()
plt.show()

'''
# initialize classifier
model_path = "cockpit_model.pth"
classifier = Classifier().to(device)

# Load model if it exists
if os.path.exists(model_path):
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded from checkpoint!")
    classifier.eval()
else:
    print("No saved model found, training from scratch.")

# Train only if needed
train_model = False  # Set to True if you want to train
if train_model:
    trainer = L.Trainer(max_epochs=2)
    trainer.fit(classifier, train_loader)
    torch.save(classifier.state_dict(), model_path)
    print("Model saved after training!")

# Testing
classifier.eval()
print(f"Testing dir : {testing_dir}")

cockpit_images = glob.glob(os.path.join(testing_dir, "Cockpit", "*.*"))
other_images = glob.glob(os.path.join(testing_dir, "Other", "*.*"))

print(f"Number of Cockpit images found: {len(cockpit_images)}")
print(f"Number of Other images found: {len(other_images)}")

test_images_paths = cockpit_images + other_images

if not cockpit_images:
    print("⚠️ Warning: No Cockpit images found in testing directory!")
if not other_images:
    print("⚠️ Warning: No Other images found in testing directory!")

def preprocess_image(image_path):
    image = read_image(image_path).float() / 255.0
    image = transforms.Resize((128, 128))(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    return image.unsqueeze(0)

y_true = []
y_score = []

image_num = range(1, len(train_losses) + 1)

plt.figure(figsize=(12,5))
#plot loss
plt.subplot(1,2,1)
plt.plot(image_num, train_losses, label='Loss')
plt.xlabel('Images')
plt.ylabel('Loss')
plt.title('Loss vs Images')
plt.legend()

#plot accuracy
plt.subplot(1, 2, 2)
plt.plot(image_num, train_accuracies, label='Accuracy', color='green')
plt.xlabel('Images')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Images')
plt.legend()

plt.tight_layout()
plt.show()

for image_path in test_images_paths:
    image = preprocess_image(image_path).to(device)
    label = 0 if 'Cockpit' in image_path else 1
    y_true.append(label)

    with torch.no_grad():
        logits = classifier(image)
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    y_score.append(prob)

y_true = np.array(y_true)
y_score = np.array(y_score).flatten()

print(f"Unique values in y_true: {set(y_true)}")
print(f"Shape of y_score: {y_score.shape}")
print(f"Shape of y_true: {y_true.shape}")
print(y_true)
print(y_score)

if len(set(y_true)) < 2:
    raise ValueError("y_true contains only one class. Ensure test images from both categories are included.")

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


#Display Sample Predictions 
plt.figure(figsize=(12, 6))
num_samples = 10
for i in range(num_samples):
    image, label = test_dataset[i]
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(image_input)
        predicted_class = logits.argmax(dim=1).item()

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(f"Pred: {predicted_class} | Actual: {label}")
    plt.axis("off")
    
plt.tight_layout()
plt.show()
'''
