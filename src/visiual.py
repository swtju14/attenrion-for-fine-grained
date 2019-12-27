import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
#import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import bilinear_cnn_all
import cv2

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = {}
        for name, module in self.submodule.module.features1._modules.items():
            if "fc" in name: 
                x = x.view(x.size(0), -1)
            
            x = module(x)
            #print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (448, 448))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature():
    pic_dir = './images/2.jpg'
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)
    print(img[0].size())
    img = img.cuda()

    net = bilinear_cnn_all.BCNN()
    net = torch.nn.DataParallel(net).cuda()
    #net = models.resnet101().to(device)
    net.load_state_dict(torch.load('./model/vgg_lamda_epoch_64.pth'))
    print(net)
    exact_list = None
    dst = './features1'
    therd_size = 448

    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        #print(type(k))
        features = v[0]
        iter_range = features.shape[0]
        #print(iter_range)
        feature_map_combination = []
        for i in range(iter_range):
            #plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            #if k not in ['0','5','10','19','28','34']:
            #    continue
            #print(features.size())
            feature = features.data.cpu().numpy()
            feature_img = feature[i,:,:]
            feature_map_combination.append(feature_img)
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            
            dst_path = os.path.join(dst, k)
            
            make_dirs(dst)
            '''
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size,therd_size), interpolation =  cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)
            '''
            dst_file = os.path.join(dst, k + '.png')
            #cv2.imwrite(dst_file, feature_img)
        feature_map_sum = sum(ele for ele in feature_map_combination)
        feature_map_sum = cv2.resize(feature_map_sum,(448,448),interpolation = cv2.INTER_NEAREST)
        img1 = img.squeeze(0).data.cpu().numpy()
        feature_map_sum = sum(i for i in [img1[0],img1[1],img1[2],feature_map_sum])
        cv2.imwrite(dst_file,feature_map_sum)
if __name__ == '__main__':
    get_feature()
