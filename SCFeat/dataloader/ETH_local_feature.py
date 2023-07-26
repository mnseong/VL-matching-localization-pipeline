import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import skimage.io as io
from path import Path
import cv2
import torch.nn.functional as F

class ETH_LFB(Dataset):
    def __init__(self, configs):
        """
        dataset for eth local feature benchmark
        """
        super(ETH_LFB, self).__init__()
        self.configs = configs
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225)),
                                             ])
        self.sift = cv2.xfeatures2d.SIFT_create()
        
        imdir       = Path(self.configs['data_path'])
        folder_dir  = imdir/self.configs['subfolder']
        images_dir  = folder_dir/'images'

        imgs = images_dir.glob('*')
        self.imfs = imgs
        self.imfs.sort()

    def __getitem__(self, item):
        imf = self.imfs[item]
        im = io.imread(imf)
        name = imf.name 
        name = '{}/{}'.format(self.configs['subfolder'], name)
        print(name)
        if len(im.shape) != 3: #gray images
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = im.copy()
        im_tensor = self.transform(im) #
        c, h, w = im_tensor.shape
        pad=(0,0,0,0)

        # now use crop to get suitable size
        crop_r = w%16
        crop_b = h%16
        im_tensor = im_tensor[:,:h-crop_b,:w-crop_r]
        im = im[:h-crop_b,:w-crop_r,:]
        # using sift keypoints
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        kpts = self.sift.detect(gray)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        coord = torch.from_numpy(kpts).float()
        im1_ori = torch.from_numpy(im)

        out = {
            'im1':      im_tensor, 
            'im1_ori':  im1_ori, 
            'coord1':   coord, 
            'name1':    name, 
            'pad1':     pad
            }

        return out

    def __len__(self):
        return len(self.imfs)