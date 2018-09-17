import torchvision
from PIL import Image
from torch.utils import data as data_utils
import glob
import os

transform = torchvision.transforms.ToTensor()

class ImageNet(data_utils.Dataset):
    """
    Creates a Pytorch Dataset for ImageNet.
    """
    def __init__(self, dir='data/'):
        self.files = glob.glob(os.path.join(dir,'*.png'))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)

    def __getitem__(self, index):
        x = Image.open(self.files[index])
        x = transform(x)
        return x