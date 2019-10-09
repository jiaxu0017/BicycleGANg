import torch.utils.data as data
from PIL import  Image
import os
import os.path
import pydicom

IMG_EXTENSIONS = ['dcm','DCM']

def is_image_file(filename):
    # 判断文件时否以IMG——EXTENSIONS结尾，即判断时否时一个医学图象文件
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir,max_dataset_size = float('inf')):
    # 查找文件中所有的指定格式的文件，并返回路径列表
    images = []
    assert  os.path.isdir(dir), '%s is not a valid directory' % dir

    for root,_ , fnames in sorted(os.walk(dir)): #os.walk输出目录中的文件名
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                images.append(path)
    return images[:min(max_dataset_size,len(images))]


def defult_loader(path):
    return pydicom.dcmread(path).pixel_array


class MedicineImageFolder(data.DataLoader):
    def __init__(self,root, transform=None, return_paths=False,
                 loader=defult_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 images in:' + root + '\n'
                               'Supported image extensions are:' +
                               ','.join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img,path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
