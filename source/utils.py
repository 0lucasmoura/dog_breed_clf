from torchvision import datasets
from torchvision import transforms

def get_validation_dataset(val_dir):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    val_dataset = datasets.ImageFolder(val_dir, 
                                       transform=transforms.Compose([
                                       transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                    ])
    return val_dataset
