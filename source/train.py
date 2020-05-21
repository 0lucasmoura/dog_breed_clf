import argparse
import json
import os
import time
import copy
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms
from model import ConvNet


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(ConvNet(model_info["hidden_dim"], model_info["output_dim"]))

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    model.to(device).eval()

    print("Done loading model.")
    return model


def _get_train_data_loader(batch_size, data_dir, num_workers):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_dir, transform=transforms.Compose([
        transforms.RandomResizedCrop(size=312, scale=(0.6, 1.0)),
        transforms.RandomRotation(10, expand=True),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader


def _get_test_data_loader(batch_size, data_dir, num_workers):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_dir, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader


def train(model, dataloaders, num_epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 30)
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for batch_X, batch_y in dataloaders[phase]:
                
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
                running_corrects += torch.sum(preds == batch_y.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start
    model.load_state_dict(best_model_wts)
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    
def save_model_params(args):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)

        
def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--output_dim', type=int, default=120, metavar='N',
                        help='number of classes of the problem')
    parser.add_argument('--hidden_dim', type=int, default=120, metavar='N',
                        help='hidden dim serving as outputs of trained models')
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='learning rate of optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='momentum of sgd optmizer')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--num-cpus', type=int, default=os.environ['SM_NUM_CPUS'])

    args = parser.parse_args()
    model_path = os.path.join(args.model_dir, 'model.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    data_loaders = {'train': _get_train_data_loader(args.batch_size, args.train_dir, args.num_cpus),
                    'val': _get_test_data_loader(args.batch_size, args.test_dir, args.num_cpus)}

    # Build the model.
    model = ConvNet(args.hidden_dim, args.output_dim).to(device)
    model = torch.nn.DataParallel(model) # recommended by sagemaker sdk python devs
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # train model
    train(model, data_loaders, args.epochs, optimizer, criterion, device)

    # Save the model and its parameters
    save_model_params(args)
    save_model(model, model_path)
