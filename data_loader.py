import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import Compose

def make_dataset(img_size, root_dir="./images"):
    dataset = ImageFolder(
        root_dir,
        transform=Compose(
            [
                RandomHorizontalFlip(),
                Resize((img_size, img_size)), 
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
            ),
    )
    return dataset

def make_data_loader(dataset:torchvision.datasets, batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        )
    return dataloader