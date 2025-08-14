import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# setting dataset for loader
class MushroomDataset(Dataset):
    def __init__(self, dataset, input_size, augment_imgs=True):
        super().__init__()

        self.dataset = dataset
        self.input_size = input_size
        self.augment = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size,input_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.RandomRotation((-30, 30))
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size,input_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.augment_imgs = augment_imgs

    def __getitem__(self, index):
        data_point, data_label = self.dataset[index]

        if self.augment_imgs:
            data_point = self.augment(data_point)
        else:
            data_point = self.transform(data_point)

        return data_point, data_label

    def __len__(self):
        return len(self.dataset)


def trainNetwork(network, optimizer, loss_fn, train_loader, device):
    true_preds, num_preds = 0., 0.
    running_loss = 0
    for train_inputs, train_labels in tqdm(train_loader):
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device).reshape(-1,1)

        preds = network(train_inputs)

        loss = loss_fn(preds, train_labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add loss to running loss
        running_loss += loss.item() * train_inputs.size(0)

        # finding and storing the train predictions to find accuracy
        with torch.no_grad():
            pred_labels = F.sigmoid(preds) > 0.5

            true_preds += (pred_labels == train_labels).sum()
            num_preds += train_labels.shape[0]

    accuracy = ((true_preds / num_preds) * 100).item()

    return accuracy, running_loss/len(train_loader)


def valNetwork(network, loss_fn, val_loader, device):
    true_preds, num_preds = 0., 0.
    running_loss = 0

    with torch.no_grad():
        for val_inputs, val_labels in tqdm(val_loader):
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device).reshape(-1,1)

            preds = network(val_inputs)

            loss = loss_fn(preds, val_labels.float())

            # add loss to running loss
            running_loss += loss.item() * val_inputs.size(0)

            # finding and storing the val predictions to find accuracy
            pred_labels = F.sigmoid(preds) > 0.5
            true_preds += (pred_labels == val_labels).sum()
            num_preds += val_labels.shape[0]

    accuracy = ((true_preds / num_preds) * 100).item()

    return accuracy, running_loss/len(val_loader)

def evalNetwork(network, test_loader, device):
    model_preds = []
    true_labels = []
    eval_info = {}

    true_preds, num_preds = 0., 0.
    with torch.no_grad():
        for test_inputs, test_labels in tqdm(test_loader):
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device).reshape(-1,1)

            preds = network(test_inputs)
            pred_labels = F.sigmoid(preds) > 0.5

            true_preds += (pred_labels == test_labels).sum()
            num_preds += test_labels.shape[0]

            model_preds += pred_labels.detach().cpu().numpy().astype(int).tolist()
            true_labels += test_labels.detach().cpu().numpy().tolist()

    accuracy = ((true_preds / num_preds) * 100).item()

    eval_info["accuracy"] = accuracy
    eval_info["model_preds"] = model_preds
    eval_info["true_labels"] = true_labels

    return eval_info


def experiment_pipeline(network, train_set, val_set, test_set, optimizer,
                        loss_fn, batch_size, epochs, input_size, device):
    # loaders
    train_loader = DataLoader(MushroomDataset(train_set, input_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MushroomDataset(val_set, input_size, augment_imgs=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MushroomDataset(test_set, input_size, augment_imgs=False), shuffle=False)

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    overfit_count = 0

    # train model
    for e in range(epochs):
        network.train()
        acc, l = trainNetwork(network, optimizer, loss_fn, train_loader, device)

        train_accs.append(acc)
        train_losses.append(l)

        network.eval()
        # validate model
        acc, l = valNetwork(network, loss_fn, val_loader, device)

        val_accs.append(acc)
        val_losses.append(l)

        print(f"Epoch-{e + 1}:")
        print(f"Train Accuracy: {train_accs[-1]}  Train Loss: {train_losses[-1]}")
        print(f"Val Accuracy: {val_accs[-1]}    Val Loss: {val_losses[-1]}")
        
        if train_accs[-1] > 90:
            if overfit(val_accs, val_losses):
                overfit_count += 1

            if overfit_count == 3:
                print("Overfit occurred 3 times, terminating training!")
                break
    

    # eval model
    network.eval()

    evals = evalNetwork(network, test_loader, device)
    evals["train_accuracies"] = train_accs
    evals["train_losses"] = train_losses
    evals["val_accuracies"] = val_accs
    evals["val_losses"] = val_losses

    plot_results(evals)

    return evals


def plot_results(evals):
    print(f"Test Accuracy: {evals['accuracy']}")

    # confusion matrix
    plt.figure()
    sns.heatmap(confusion_matrix(evals["true_labels"], evals["model_preds"]),
                annot=True, fmt=".0f", cmap="Blues",
                annot_kws={"size": 15},
                xticklabels=["edible", "poisonous"],
                yticklabels=["edible", "poisonous"])
    plt.title("Confusion Matrix")
    plt.xlabel("Preds")
    plt.ylabel("Labels")

    plt.show()


# PRE-TRAINED MODELS
class AlexNetBackBoned(nn.Module):
    def __init__(self):
        super(AlexNetBackBoned, self).__init__()
        # Load pretrained alexnet
        self.alexnet = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", weights="DEFAULT")

        # freezing the alexnet parameters
        for param in self.alexnet.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.alexnet(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class VGG19BNBackBoned(nn.Module):
    def __init__(self):
        super(VGG19BNBackBoned, self).__init__()
        # Load pretrained vgg19_bn
        self.vgg19_bn = torch.hub.load("pytorch/vision:v0.10.0", "vgg19_bn", weights="DEFAULT")

        # freezing the vgg parameters
        for param in self.vgg19_bn.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.vgg19_bn(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class DenseNetBackBoned(nn.Module):
    def __init__(self):
        super(DenseNetBackBoned, self).__init__()
        # Load pretrained densenet
        self.densenet = torch.hub.load("pytorch/vision:v0.10.0", "densenet161", weights="DEFAULT")

        # freezing the densenet parameters
        for param in self.densenet.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.densenet(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class ResnetBackBoned(nn.Module):
    def __init__(self):
        super(ResnetBackBoned, self).__init__()
        # Load pretrained resnet
        self.resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", weights="DEFAULT")

        # freezing the resnet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class MobileNetV3BackBoned(nn.Module):
    def __init__(self):
        super(MobileNetV3BackBoned, self).__init__()
        # Load pretrained mobilenet
        self.mobilenet = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v3_large", weights="DEFAULT")## VGG19-BN

        # freezing the mobilenet parameters
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm1d(1000)
        self.linear1 = torch.nn.Linear(1000, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

