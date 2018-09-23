import torchvision
from PIL import Image
from torch.utils import data as data_utils
import glob
import os

transform = torchvision.transforms.ToTensor()

class customLoader(data_utils.Dataset):
    def __init__(self, dir='data/'):
        self.files = glob.glob(os.path.join(dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x = open(self.files[index])
        return x