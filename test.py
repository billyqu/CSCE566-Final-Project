import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from train import NPZDataset, device  # Import custom dataset class and device
from sklearn.metrics import roc_auc_score
# Define paths
model_path = "/home/billyqu/CSCE566_final_project/model/vgg_best_model.pth"
test_data_path = "/home/billyqu/CSCE566_final_project/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/test"

# Define preprocessing for the test dataset
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),                    # Convert to Tensor
    transforms.Resize((224, 224)),            # Resize to VGG input size
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load the test dataset and DataLoader
test_dataset = NPZDataset(test_data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the same model architecture as in training
num_classes = len(np.unique([label for _, label, _ in test_dataset]))

vgg_model = models.vgg16(pretrained=True)
for param in vgg_model.features.parameters():
    param.requires_grad = True
vgg_model.classifier[-1] = nn.Linear(vgg_model.classifier[-1].in_features, 2)
print("loading vgg model")
vgg_model.load_state_dict(torch.load(model_path))
vgg_model = vgg_model.to(device)
vgg_model.eval()

# densenet_model = models.densenet121(pretrained=True)
# for param in densenet_model.features.parameters():
#     param.requires_grad = True
# densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 2)
# print("loading densenet model")
# densenet_model.load_state_dict(torch.load(model_path))
# densenet_model = densenet_model.to(device)
# densenet_model.eval()

# Define the evaluation function
def evaluate_with_auc(model, loader):
    all_predictions = []
    all_labels = []
    all_genders = []

    with torch.no_grad():
        for images, labels, genders in tqdm(loader, unit="batch", desc="Testing"):
            images, labels, genders = images.to(device), labels.to(device), genders.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_genders.extend(genders.cpu().numpy())

    overall_auc = roc_auc_score(all_labels, all_predictions)
    male_auc = roc_auc_score(
        [all_labels[i] for i in range(len(all_labels)) if all_genders[i] == 0],
        [all_predictions[i] for i in range(len(all_predictions)) if all_genders[i] == 0]
    )
    female_auc = roc_auc_score(
        [all_labels[i] for i in range(len(all_labels)) if all_genders[i] == 1],
        [all_predictions[i] for i in range(len(all_predictions)) if all_genders[i] == 1]
    )

    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Male AUC: {male_auc:.4f}")
    print(f"Female AUC: {female_auc:.4f}")
    return overall_auc, male_auc, female_auc

# Run evaluation
evaluate_with_auc(vgg_model, test_loader)

