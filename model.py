import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

##https://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(filter_size, weights):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    f_out = weights.size(0)
    f_in = weights.size(1)
    weights = np.zeros((f_out,
                        f_in,
                        4,
                        4), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in xrange(f_out):
        for j in xrange(f_in):
            weights[i, j, :, :] = upsample_kernel
    return torch.Tensor(weights)        


    
class FeatureExtraction(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction, self).__init__()
        if level==1:
            self.conv0 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))
        else:
            self.conv0 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.convt_F = nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1))
        self.LReLus = nn.LeakyReLU(negative_slope=0.2)
        self.convt_F.weight.data.copy_(bilinear_upsample_weights(4, self.convt_F.weight))
       
    def forward(self, x):
        out = self.LReLus(self.conv0(x))
        out = self.LReLus(self.conv1(out))
        out = self.LReLus(self.conv2(out))
        out = self.LReLus(self.conv3(out))
        out = self.LReLus(self.conv4(out))
        out = self.LReLus(self.conv5(out))
        out = self.LReLus(self.convt_F(out))
        return out


class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv_R = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))
        self.convt_I = nn.ConvTranspose2d(1, 1, (4, 4), (2, 2), (1, 1))
        self.convt_I.weight.data.copy_(bilinear_upsample_weights(4, self.convt_I.weight))     
        
    def forward(self, LR, convt_F):
        convt_I = self.convt_I(LR)
        conv_R = self.conv_R(convt_F)
        
        HR = convt_I+conv_R
        return HR
        
        
class LasSRN(nn.Module):
    def __init__(self):
        super(LasSRN, self).__init__()
        self.FeatureExtraction1 = FeatureExtraction(level=1)
        self.FeatureExtraction2 = FeatureExtraction(level=2)
        self.FeatureExtraction3 = FeatureExtraction(level=3)
        self.ImageReconstruction1 = ImageReconstruction()
        self.ImageReconstruction2 = ImageReconstruction()
        self.ImageReconstruction3 = ImageReconstruction()



    def forward(self, LR):
        
        convt_F1 = self.FeatureExtraction1(LR)
        HR_2 = self.ImageReconstruction1(LR, convt_F1)
        
        convt_F2 = self.FeatureExtraction2(convt_F1)
        HR_4 = self.ImageReconstruction2(HR_2, convt_F2)
        
        convt_F3 = self.FeatureExtraction3(convt_F2)
        HR_8 = self.ImageReconstruction3(HR_4, convt_F3)
        
        return HR_2, HR_4, HR_8

        

    