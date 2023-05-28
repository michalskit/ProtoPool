import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from pathlib import Path

from custom_model import Resnet50iNaturalist


def adjust_learning_rate(optimizer, rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= rate

def set_trainable_attr(m, b=True):
    for p in m.parameters(): p.requires_grad = b

def unfreeze(model, l):
    features = list(model.features.named_children())
    set_trainable_attr(features[l][1])

def freeze_all_but_last_layer(model):
    for param in model.features.parameters():
        param.requires_grad = False

def unfreeze_three_layers(model):
    unfreeze(model, 7)
    unfreeze(model, 6)
    unfreeze(model, 5)

def unfreeze_three_more(model):
    unfreeze(model, 4)
    unfreeze(model, 3)
    unfreeze(model,2)

def save_model(model, path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }, path)

def custom_learn_model(lr=0.0005, num_epochs = 50, in_dev_mode=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')

    model = Resnet50iNaturalist(num_classes=100)
    model.to(device)

    # Load the datasets
    transforms_train_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if in_dev_mode:
        train_path = '/home/robin/data/selected/CUB_200_2011/train_cropped_augmented/'
        test_path = '/home/robin/data/selected/CUB_200_2011/test_cropped/'
    else:
        train_path = '/home/robin/d/CUB_200_2011/train_cropped_augmented_in/'
        test_path = '/home/robin/d/CUB_200_2011/test_cropped_in/'

    train_dataset = datasets.ImageFolder(train_path, transform=transforms_train_test)
    train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=4)

    test_dataset = datasets.ImageFolder(test_path, transform=transforms_train_test)
    test_loader = DataLoader(test_dataset, batch_size=80, shuffle=False, num_workers=4)
        
    seed = np.random.randint(10, 10000, size=1)[0]
    torch.manual_seed(seed)

    optimizer_last_layer = torch.optim.Adam(
        [{'params': model.last_layer.parameters(), 'lr': 3 * lr, 'weight_decay': 1e-3}])

    optimizer_three_layers = torch.optim.Adam(
        [{'params': model.features.parameters(), 'lr': lr / 10, 'weight_decay': 1e-3},
        {'params': model.last_layer.parameters(), 'lr': 3 * lr, 'weight_decay': 1e-3}])

    optimizer_all_layers = torch.optim.Adam(
        [{'params': model.features.parameters(), 'lr': lr / 10, 'weight_decay': 1e-3},
        {'params': model.last_layer.parameters(), 'lr': 3 * lr, 'weight_decay': 1e-3}])

    criterion = torch.nn.CrossEntropyLoss()

    info =  'resnet50_pretrained_on_iNaturalist' \
            f'_lr-{lr}' \
            f'_seed-{seed}' \
            f'_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}'

    path_tensorboard = f'../results_analysis/tensorboard/{info}'
    Path(path_tensorboard).mkdir(parents=True, exist_ok=True)
    dir_checkpoint = f'../results_analysis/checkpoint/{info}'
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(path_tensorboard)

    best_test_accuracy = 0.0
    loss_not_improved_for_epochs = 0
    best_test_loss = float('inf')

    epoch_tqdm = range(0, num_epochs)

    for epoch in epoch_tqdm:

        if epoch == 0:
            freeze_all_but_last_layer(model)
            print('\nFreezing all but last layer\n')
        elif epoch == 10:
            unfreeze_three_layers(model)
            print('\nUnfreezing 3 layers from the end\n')
        elif epoch == 30:
            print('\nUnfreezing three more layers\n')
            unfreeze_three_more(model)


        if epoch < 10:
            optimizer = optimizer_last_layer
        elif epoch < 30:
            optimizer = optimizer_three_layers
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        else:
            optimizer = optimizer_all_layers
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels) # + l1 norm ?
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


            writer.add_scalar('train/batch_loss', train_loss,
                        epoch * len(train_loader) + i)

        if epoch >= 10:
            lr_scheduler.step()

        train_epoch_loss = train_loss / len(train_loader.dataset)
        train_epoch_acc = 100 * correct / total
        print(f'Train Loss: {train_epoch_loss:.4f}, Acc: {train_epoch_acc:.2f}%')

        writer.add_scalar('train/loss', train_loss,
                        epoch)
        writer.add_scalar('train/acc', train_epoch_acc,
                        epoch)

        # Validation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        for images, labels in test_loader:
            inputs, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss.item() * inputs.size(0)

        test_epoch_loss = test_loss / len(test_loader.dataset)
        test_epoch_acc = 100 * correct / total
        print(f'Test  Loss: {test_epoch_loss:.4f}, Acc: {test_epoch_acc:.2f}%')

        writer.add_scalar('test/loss', test_epoch_loss, epoch)
        writer.add_scalar('test/acc', test_epoch_acc, epoch)
        
        if (test_epoch_acc > best_test_accuracy) or (test_epoch_loss < best_test_loss):
            loss_not_improved_for_epochs = 0
            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
            # save the best model
            if test_epoch_acc > best_test_accuracy:
                best_test_accuracy = test_epoch_acc
                save_model(model, f'{dir_checkpoint}/best_model.pth', epoch)
        else:
            loss_not_improved_for_epochs += 1

        if loss_not_improved_for_epochs >= 5:
            adjust_learning_rate(optimizer, 0.95)

        # I don't use early stopping

