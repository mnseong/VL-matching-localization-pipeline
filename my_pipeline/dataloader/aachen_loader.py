import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from path import Path
import cv2
import torch.nn.functional as F

# configs = {
#     'data_path':/mnt/hdd4T/minseong/github/VL-matching-localization-pipeline/datasets/Aachen-Day-Night-v1.1/,
# }

data_path = "/mnt/hdd4T/minseong/github/VL-matching-localization-pipeline/datasets/Aachen-Day-Night-v1.1/"

class Aachen_Day_Night(Dataset):
    def __init__(self):
        super(Aachen_Day_Night, self).__init__()
        self.path = data_path
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ])
        self.sift = cv2.xfeatures2d.SIFT_create()

        imdir       = Path(self.path)
        dbimgs      = imdir.glob('mapping/sensors/records_data/db/*.jpg')
        queryimgs   = imdir.glob('query/*/*/*/*/*/*.jpg')
        sequences1  = imdir.glob('mapping/sensors/records_data/sequences/gopro3_undistorted/*.png')
        sequences2  = imdir.glob('mapping/sensors/records_data/sequences/nexus4_sequences/*/*.png')
        
        self.imfs = dbimgs
        self.imfs.extend(queryimgs)
        self.imfs.extend(sequences1)
        self.imfs.extend(sequences2)
        self.imfs.sort()


    def __getitem__(self, item):
        imf = self.imfs[item]
        im = io.imread(imf)
        imf_split = imf.split('/')
        if 'db' in imf_split:
            name = imf_split[-2:]
            name = '/'.join(name)
        elif 'query' in imf_split:
            name = imf_split[-4:]
            name = '/'.join(name)
        elif 'gopro3_undistorted' in imf_split:
            name = imf_split[-3:]
            name = '/'.join(name)
        elif 'nexus4_sequences' in imf_split:
            name = imf_split[-4:]
            name = '/'.join(name)
        im_tensor = self.transform(im)
        c, h, w = im_tensor.shape
        pad=(0,0,0,0)

        # now use crop to get suitable size
        crop_r = w%16
        crop_b = h%16
        im_tensor = im_tensor[:,:h-crop_b,:w-crop_r]
        im = im[:h-crop_b,:w-crop_r,:]
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        kpts = self.sift.detect(gray)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        coord = torch.from_numpy(kpts).float()
        im1_ori = torch.from_numpy(im)
        out = {'im1': im_tensor, 'im1_ori':im1_ori, 'coord1': coord, 'name1': name, 'pad1':pad}
        return out

    def __len__(self):
        return len(self.imfs)