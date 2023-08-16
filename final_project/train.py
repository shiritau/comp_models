import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_datasets
import numpy as np
from clearml import Task
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def get_efficientnet_model():
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                  'nvidia_efficientnet_b0')  # create new classifier with sigmoid activation function
    model = nn.Sequential(
        efficientnet,
        nn.Linear(in_features=1000, out_features=1),
        nn.Sigmoid()
    )
    return model


def get_vgg_model():
    vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights='DEFAULT').eval()
    vgg.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=1)
    # create new classifier with sigmoid activation function
    model = nn.Sequential(
        vgg,
        nn.Sigmoid()
    )
    return model


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        output = self.sigmoid(x)
        return output


def calc_sens_spec(output, label, accuracy_threshold):
    eps = 1e-7
    tp = ((output > accuracy_threshold) & (label == 1)).sum().item()
    tn = ((output < accuracy_threshold) & (label == 0)).sum().item()
    fp = ((output > accuracy_threshold) & (label == 0)).sum().item()
    fn = ((output < accuracy_threshold) & (label == 1)).sum().item()
    sensitivity = tp / (tp + fn+eps)
    specificity = tn / (tn + fp+eps)
    return sensitivity, specificity


def train(df_path, model, num_epochs, weights_dir, exp_name, accuracy_threshold=0.5):
    train_dataset, val_dataset = get_datasets(df_path)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    writer = SummaryWriter(log_dir=weights_dir, comment=exp_name)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_train_sens = []
        epoch_train_spec = []
        model.train()
        for batch_idx, data in enumerate(train_dataloader):
            if batch_idx >= 50:
                break
            # data = data.view(-1, 28 * 28)
            scan = data['scan'].to(torch.float32)
            scan = scan.view(-1, 240 * 240)
            label = data['label'].unsqueeze(1).to(torch.float32)
            optimizer.zero_grad()
            output = model(scan)
            loss = criterion(output, label)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            accuracy = ((output > accuracy_threshold) == label).sum().item() / len(label)
            epoch_train_acc.append(accuracy)
            sensitivity, specificity = calc_sens_spec(output, label, accuracy_threshold)
            epoch_train_sens.append(sensitivity)
            epoch_train_spec.append(specificity)

            if batch_idx % 10 == 0:
                print('Train Epoch: {}, iteration: {}, Loss: {:.6f}, Acc: {:.6f}'.format(epoch, batch_idx,
                                                                                         np.mean(epoch_train_loss),
                                                                                         np.mean(epoch_train_acc)))
        writer.add_scalar('Loss/train', np.mean(epoch_train_loss), epoch)
        writer.add_scalar('Accuracy/train', np.mean(np.array(epoch_train_acc)), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Sensitivity/train', np.mean(np.array(epoch_train_sens)), epoch)
        writer.add_scalar('Specificity/train', np.mean(np.array(epoch_train_spec)), epoch)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            epoch_val_loss = []
            epoch_val_acc = []
            for batch_idx, data in enumerate(val_dataloader):
                if batch_idx >= 100:
                    break
                scan = data['scan'].to(torch.float32)
                scan = scan.view(-1, 240 * 240)
                label = data['label'].unsqueeze(1).to(torch.float32)
                output = model(scan)
                loss = criterion(output, label)
                epoch_val_loss.append(loss.item())
                accuracy = ((output > accuracy_threshold) == label).sum().item() / len(label)
                sensitivity, specificity = calc_sens_spec(output, label, accuracy_threshold)
                epoch_train_sens.append(sensitivity)
                epoch_train_spec.append(specificity)
                epoch_val_acc.append(accuracy)
            print('Val Epoch: {}, Loss: {:.6f}, Acc: {:.6f}'.format(epoch, np.mean(epoch_val_loss),
                                                                    np.mean(epoch_val_acc)))

        writer.add_scalar('Loss/val', np.mean(epoch_val_loss), epoch)
        writer.add_scalar('Accuracy/val', np.mean(np.array(epoch_val_acc)), epoch)
        writer.add_scalar('Sensitivity/val', np.mean(np.array(epoch_train_sens)), epoch)
        writer.add_scalar('Specificity/val', np.mean(np.array(epoch_train_spec)), epoch)

    return model, val_dataset


def test_roc(model, test_dataset):
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    outputs = []
    labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            scan = data['scan'].to(torch.float32)
            scan = scan.view(-1, 240 * 240)
            label = data['label'].unsqueeze(1).to(torch.float32)
            output = model(scan)
            outputs.append(output.item())
            labels.append(label.item())
    fpr, tpr, thresholds = roc_curve(labels, outputs)
    auc_score = roc_auc_score(labels, outputs)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    ROOT_DIR = r"C:\Users\shiri\Documents\School\Master\Courses\Comp_models_of_learning\final_project\brats"
    WEIGHTS_DIR = r"C:\Users\shiri\Documents\School\Master\Courses\Comp_models_of_learning\final_project\brats\exps"
    DF_PATH = r"C:\Users\shiri\Documents\School\Master\Courses\Comp_models_of_learning\final_project\brats\paths.csv"
    NUM_EPOCHS = 30
    ACCURACY_THRESHOLD = 0.5

    model = MLP(240 * 240, 50, 1)
    #model = get_efficientnet_model()
    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    task = Task.init(project_name="tumour_detection", task_name=f"lr_0.01_{current_time}")

    model, test_dataset = train(DF_PATH, model, NUM_EPOCHS, WEIGHTS_DIR, task.name, ACCURACY_THRESHOLD)
    test_roc(model, test_dataset)
