import torchvision.transforms as T


def get_transforms(phase="train"):
    """
    Returns the data transformation pipeline based on the phase (train/val/test).

    Parameters:
    - phase: str, one of ["train", "val", "test"]

    Returns:
    - transforms: torchvision.transforms.Compose object
    """
    if phase == "train":
        transforms = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transforms = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms