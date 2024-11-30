import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset for `.npz` files
class NPZDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data["slo_fundus"]
        label = data["dr_class"]
        gender = data["male"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(gender, dtype=torch.long)


# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Function to initialize the VGG16 model
def initialize_model(num_classes):
    # vgg_model = models.vgg16(pretrained=True)  # Load the pre-trained model
    # for param in vgg_model.features.parameters():
    #     param.requires_grad = True  # Fine-tune the features

    # # Replace the classifier's final layer with one for our task
    # vgg_model.classifier[-1] = nn.Linear(vgg_model.classifier[-1].in_features, num_classes)
    # return vgg_model.to(device)

    densenet_model = models.densenet121(pretrained=True)  # Load pre-trained DenseNet
    for param in densenet_model.features.parameters():
        param.requires_grad = True  # Fine-tune the features

    # Replace the classifier with a new one for the current task
    densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, num_classes)
    return densenet_model.to(device)
# Training loop
def train_model(model, train_loader, val_loader, num_epochs=20, save_path="best_model.pth"):
    criterion = nn.CrossEntropyLoss()  # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Define optimizer

    best_auc = 0.0  # Track the best validation accuracy
    global auc_history

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels, _ in tqdm(train_loader, unit="batch", desc="Training"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1} - Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validate the model
        val_accuracy, overall_auc, male_auc, female_auc = validate_model(model, val_loader)

        auc_history["epoch"].append(epoch + 1)
        auc_history["overall_auc"].append(overall_auc)
        auc_history["male_auc"].append(male_auc)
        auc_history["female_auc"].append(female_auc)

        # Save the model if validation accuracy improves
        if overall_auc > best_auc:
            best_auc = overall_auc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with Overall AUC: {best_auc:.4f}")

# Validation loop
from sklearn.metrics import roc_auc_score

def validate_model(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_genders = []
    criterion = nn.CrossEntropyLoss()  # Define loss function
    with torch.no_grad():
        for images, labels, genders in tqdm(val_loader, unit="batch", desc="Validating"):
            images, labels, genders = images.to(device), labels.to(device), genders.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_genders.extend(genders.cpu().numpy())

    val_accuracy = 100. * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Compute AUCs
    overall_auc = roc_auc_score(all_labels, all_predictions)
    male_auc = roc_auc_score(
        [all_labels[i] for i in range(len(all_labels)) if all_genders[i] == 0],
        [all_predictions[i] for i in range(len(all_predictions)) if all_genders[i] == 0],
    )
    female_auc = roc_auc_score(
        [all_labels[i] for i in range(len(all_labels)) if all_genders[i] == 1],
        [all_predictions[i] for i in range(len(all_predictions)) if all_genders[i] == 1],
    )

    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Male AUC: {male_auc:.4f}")
    print(f"Female AUC: {female_auc:.4f}")

    return val_accuracy, overall_auc, male_auc, female_auc

if __name__ == "__main__":
    # Paths to data folders
    train_path = "/home/billyqu/CSCE566_final_project/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/train"
    val_path = "/home/billyqu/CSCE566_final_project/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/val"

    # Initialize datasets and dataloaders
    train_dataset = NPZDataset(train_path, transform=transform)
    val_dataset = NPZDataset(val_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Determine the number of classes
    num_classes = len(np.unique([label for _, label, _ in train_dataset]))

    auc_history = {"epoch": [], "overall_auc": [], "male_auc": [], "female_auc": []}

    # Initialize and train the model
    model = initialize_model(num_classes)
    train_model(model, train_loader, val_loader, num_epochs=20)

    plt.figure(figsize=(10, 6))
    plt.plot(auc_history["epoch"], auc_history["overall_auc"], label="Overall AUC")
    plt.plot(auc_history["epoch"], auc_history["male_auc"], label="Male AUC")
    plt.plot(auc_history["epoch"], auc_history["female_auc"], label="Female AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.xticks(auc_history["epoch"])
    plt.savefig("densenet_auc_vs_epoch.png")  # Save the plot
    plt.show()

    highest_overall_auc = max(auc_history["overall_auc"])
    highest_male_auc = max(auc_history["male_auc"])
    highest_female_auc = max(auc_history["female_auc"])
    print(f"Highest Overall AUC: {highest_overall_auc:.4f}")
    print(f"Highest Male AUC: {highest_male_auc:.4f}")
    print(f"Highest Female AUC: {highest_female_auc:.4f}")