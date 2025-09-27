import os.path as osp # use this for dataset loading path
from torch.utils.data import DataLoader
import PIL.Image as Image

# first load the local images into a DataFolder
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms

def loader_function(path: str):
    """
    define a helper function to load images in DatasetFolder
    """
    with open(path, "rb") as f:
        image = Image.open(f).convert("RGB")
        return image

def build_dataset(data_path: str, dataset_name: str, resolution: int):
    """
    define dataset loading utils function
    """
    
    # define training augmentations and validation augmentation
    train_aug_method = [
        transforms.Resize(resolution),
        transforms.ToTensor(),
    ]

    val_aug_method = [
        transforms.Resize(resolution), 
        transforms.ToTensor(),
    ]

    train_aug, val_aug = transforms.Compose(train_aug_method), transforms.Compose(val_aug_method)

    if dataset_name == "imagenet":
        train_dataset = DatasetFolder(root = osp.join(data_path, "train"), loader=loader_function, extensions=IMG_EXTENSIONS, transform=train_aug)
        val_dataset = DatasetFolder(root = osp.join(data_path,"val"), loader=loader_function, extensions=IMG_EXTENSIONS, transform=val_aug)
    else:
        raise NotImplementedError("This dataset has not been supported yet")
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="local path of dataset")
    parser.add_argument("--dataset_name", type=str, help="name of the dataset")
    parser.add_argument("--resolution", type=int, help="image resize resolution")
    args = parser.parse_args()

    train_dataset, val_dataset = build_dataset(args.dataset_path, args.dataset_name, args.resolution)
    print(f"this is the sample number of train dataset: {len(train_dataset)}")