import torch
from torchvision import datasets, transforms
import os


def save_ds_csv(ds, filename):
    if os.path.exists(filename):
        print(filename + " done")
        return

    xs = []
    ys = []

    with open(filename, 'w') as f:
        for i in range(len(ds)):
            x, y = ds[i]
            x = x.view((-1))
            for el in x:
                f.write("{0:.4f}".format(el.item()))
                f.write(",")
            f.write(str(y))
            f.write("\n")

    print(filename + " done")


def main():
    train_ds = datasets.SVHN(
        root='.', split='train', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    save_ds_csv(train_ds, 'train_ds')

    test_ds = datasets.SVHN(
        root='.', split='test', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    save_ds_csv(test_ds, 'test_ds')


if __name__ == "__main__":
    main()
