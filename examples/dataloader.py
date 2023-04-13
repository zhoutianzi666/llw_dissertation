import os
import numpy as np


from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2

# Your Data Path
img_dir = '/zhoukangkang/Paddle_2_3/PaddleTest/inference/python_api_test/test_int8_model/dataset/ILSVRC2012_val/val'
anno_file = "/zhoukangkang/Paddle_2_3/PaddleTest/inference/python_api_test/test_int8_model/dataset/ILSVRC2012_val/val_list.txt"

class MyDataset(Dataset):
    def __init__(self, img_dir, anno_file, imgsz=(640, 640)):
        self.img_dir = img_dir
        self.anno_file = anno_file
        self.imgsz = imgsz
        # self.img_namelst 就是图片名字的列表
        self.img_namelst = os.listdir(self.img_dir)
        print(self.img_namelst)

    # need to overload
    def __len__(self):
        return len(self.img_namelst)

    # need to overload
    def __getitem__(self, idx):
        with open(self.anno_file, 'r') as f:
            label = f.readline().strip()
        img = cv2.imread(os.path.join(img_dir, self.img_namelst[idx]))
        img = cv2.resize(img, self.imgsz)
        return img, label


dataset = MyDataset(img_dir, anno_file)
dataloader = DataLoader(dataset=dataset, batch_size=1)

# display
for img_batch, label_batch in dataloader:
    img_batch = img_batch.numpy()
    print(img_batch.shape)
    # img = np.concatenate(img_batch, axis=0)
    if img_batch.shape[0] == 2:
        img = np.hstack((img_batch[0], img_batch[1]))
    else:
        img = np.squeeze(img_batch, axis=0)  # 最后一张图时，删除第一个维度
    print(img.shape)
    break
    #cv2.imshow(label_batch[0], img)
    #cv2.waitKey(0)

