import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DogCatDataset(Dataset):
    def __init__(self, dir, transform = None):
        self.dir = dir
        self.transform = transform
        self.images = os.listdir(self.dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.dir, self.images[index])
        label = self.images[index].split(".")[0]
        label = 0 if label == 'cat' else 1
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    train_dir = 'data/train'
    dataset = DogCatDataset(dir=train_dir)

    first_five = [dataset[i] for i in range(5)]
    last_five = [dataset[i] for i in range(len(dataset) - 5, len(dataset))]

    print("First 5:")
    for i, (img, label) in enumerate(first_five):
        print(f"  {i}: (img.shape={img.shape}, label={label})")

    print("\nLast 5:")
    for i, (img, label) in enumerate(last_five):
        print(f"  {len(dataset) - 5 + i}: (img.shape={img.shape}, label={label})")