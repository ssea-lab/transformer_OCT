import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from util import ImageProcessor
from tifffile import TiffFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OCTDataSet(Dataset):
    def __init__(self, args, folder_num=0, mode='train', augmentation='supervised_train'):
        assert (mode in ['ssl', 'train', 'val', 'test']), "mode must be type in ['ssl','train','test','val']"
        assert (augmentation in ['supervised_train', 'val_or_test',
                                 'dino_ssl_train', 'dino_fine_tune_train',
                                 'mae_ssl_train', 'mae_fine_tune_train',
                                 ]), \
            "augmentation must be type in ['supervised_train','val_or_test'," \
            "'dino_ssl_train','dino_fine_tune_train', 'mae_ssl_train', 'mae_fine_tune_train']"

        fileNames = self.__load_file(mode=mode, folder_num=folder_num)
        self.imgs = []
        self.labels = []
        self.img_processor = ImageProcessor(args, augmentation=augmentation)
        for fileName in fileNames:
            self.imgs.append(fileName.split('\t')[0])
            self.labels.append(int(fileName.split('\t')[1]))

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        label = self.labels[index]
        img = Image.open(imgPath).convert('RGB')  # [H, W, 3]
        img = self.img_processor(img)  # [3, H, W]
        return [img, label, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pre="data_folder", mode='train', folder_num=0):
        if mode in ['train', 'val']:
            mode += '_folder_' + str(folder_num) + '.txt'
        elif mode == 'ssl':
            mode = 'self_train.txt'
        else:
            mode += '_folder.txt'

        mode = os.path.join(pre, mode)
        fileNames = []
        with open(mode, 'r') as file:
            lines = file.readlines()
            for line in lines:
                fileNames.append(line.strip("\n"))
        return fileNames


class TiffDataSet(Dataset):
    def __init__(self, args, tiff_file):
        self.args = args
        self.imgs, self.labels = self.__load_file(tiff_file)
        self.img_processor = ImageProcessor(args, augmentation='val_or_test')

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        label = self.labels[index]
        self.tiff_patch_list = self.split_tiff_to_patches(imgPath)  # list of patches [(H, W, 3)]
        for i in range(len(self.tiff_patch_list)):
            self.tiff_patch_list[i] = self.img_processor(self.tiff_patch_list[i])  # list of patches [(3, H, W)]
        if not self.args.ensemble:
            img = torch.stack(self.tiff_patch_list, dim=0)  # [60, 3, H, W]
        else:
            img1, img2, img3 = [], [], []
            for tiff_patch in self.tiff_patch_list:
                img1.append(tiff_patch[0])
                img2.append(tiff_patch[1])
                img3.append(tiff_patch[2])
            img1 = torch.stack(img1, dim=0)  # [60, 3, H, W]
            img2 = torch.stack(img2, dim=0)  # [60, 3, H, W]
            img3 = torch.stack(img3, dim=0)  # [60, 3, H, W]
            img = [img1, img2, img3]

        return [img, label, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, tiff_file):
        images = []
        labels = []
        with open(tiff_file, 'r', encoding='utf8') as fin:
            for line in fin:
                image = line.strip('\n').split('\t')[0]
                label = line.strip('\n').split('\t')[1]
                images.append(image)
                labels.append(int(label))
        return images, labels

    def findEdge(self, tiff_arr):
        radius = 5
        edgeI = 50
        edge = -1
        lineMean = np.mean(tiff_arr[0, :, :], axis=-1)
        for idx in range(100, 500):
            if np.mean(lineMean[idx - 5:idx + 5]) >= 50:
                edge = idx
                break
        if edge < 100 or edge > 500:
            edge = 500
        return edge

    def split_tiff_to_patches(self, tiff_file_name):

        tif = TiffFile(tiff_file_name)
        edge = self.findEdge(tif.asarray())
        image_arr = tif.asarray()
        original_im = image_arr[:, edge - 75:, :]
        tif_patch_list = []
        # 一张tiff文件中包含10帧，逐帧循环，逐帧切分成patch
        for i in range(len(original_im)):
            image = Image.fromarray(original_im[i], 'L')

            img = image
            height = img.size[1]
            width = img.size[0]
            # width=600,step=100
            step_interval = 100
            step = math.ceil((width - 600) / step_interval) + 1

            for j in range(step):
                img_crop = image.crop((j * step_interval, 0, j * step_interval + 600, height))
                img_crop = img_crop.convert('RGB')
                tif_patch_list.append(img_crop)
        return tif_patch_list

