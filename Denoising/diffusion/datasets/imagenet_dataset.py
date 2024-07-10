import os
from torch.utils.data import Dataset
from PIL import Image
import json

# https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f) # dict example: 0: ['n214454132135', 'goldfish']
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id) # folder_name > num_class
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in sorted(os.listdir(samples_dir)):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch
    import torchvision
    from tqdm import tqdm
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std),
        ]
    )

    dataset = ImageNetKaggle( '/data1/erez/datasets/imagenet', "val", val_transform)
    x, t = dataset[1]
    print(x.shape, t)
    torchvision.utils.save_image(x, 'valimg0.png')
    pass
if __name__ == '__mainOrig__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch
    import torchvision
    from tqdm import tqdm

    model = torchvision.models.resnet50(weights="DEFAULT")
    model.eval().cuda()  # Needs CUDA, don't bother on CPUs
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = ImageNetKaggle( '< YOUR_FOLDER >', "val", val_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=64,  # may need to reduce this depending on your GPU
        num_workers=8,  # may need to reduce this depending on your num of CPUs and RAM
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            y_pred = model(x.cuda())
            correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
            total += len(y)
    print(correct / total)

# the correct class list: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a / https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/

# other approach: https://github.com/pytorch/examples/tree/main/imagenet