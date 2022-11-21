import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)

def load_sample(path):
    with open(path,'r') as f:
        data = f.readlines()
    A = torch.zeros(4,56)
    for i in range(len(data)):
        arr=data[i].split(' ')
        B = arr[1::2]
        k = 0
        for j in range(len(B)):
            A[i][k] = int(B[j])
            k = k + 1
    return A

def load_point(path,cls):
    with open(path,'r') as f:
        data = f.readlines()
    x = torch.zeros(4,56)
    y = torch.zeros(4,2)
    m = 0
    for i in range(len(data)):
        arr = data[i].split(' ')
        y_tmp = arr[1::2]
        x_tmp = arr[::2]
        for n in range(m,4):
            if cls[n] != 0:
                for j in range(len(y_tmp)):
                    x[n][j] = float(x_tmp[j])
#                    y[n][j] = float(y_tmp[j])
                    y[n][0] = int(y_tmp[0])
                    y[n][1] = int(y_tmp[len(y_tmp)-1])
                break
        m = n + 1
#    x = x / 1280
#    y = y / 720
#    x = torch.cat((torch.zeros(1,56),x),dim=0)
#    y = torch.cat((torch.zeros(1,2),y),dim=0)
    return x,y

class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane


    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
#        label_path = img_path.replace('.jpg','.lines.txt')
        img = loader_func(img_path)
#        label = load_sample(label_path)
#        print(label)
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform = None,target_transform = None,simu_transform = None, load_name = False,
                row_anchor = None,use_aux=False,segment_transform=None, num_lanes = 4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()
            self.point_files = [path.replace('.jpg','.lines.txt') for path in self.list]
        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_point = self.point_files[index]
#        print(len(self.point_files))
        l_info = l.split()
        l_point_info  = l_point.split()
        img_name, label_name = l_info[0], l_info[1]
        cls_label = torch.zeros(4)
        cls_label[0] = int(l_info[2])
        cls_label[1] = int(l_info[3])
        cls_label[2] = int(l_info[4])
        cls_label[3] = int(l_info[5])
        point_name = l_point_info[0]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]
            point_name = point_name[1:]
        
        label_path = os.path.join(self.path, label_name)
        point_path = os.path.join(self.path, point_name)
        label = loader_func(label_path)
#        print(point_path)
        label_x,label_y = load_point(point_path,cls_label)
#        print(label_y)
        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)
    

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
#        lane_pts = self._get_index(label)
        # get the coordinates of lanes at row anchors

#        print(cls_label)
      
        w, h = img.size
#        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
#            return img, cls_label, seg_label
#            print(img.shape)
            return img, cls_label,seg_label,label_x,label_y
           # return img, seg_label
        if self.load_name:
#            return img, cls_label, img_name
            return img, cls_label,img_name
#        print(img.size, cur_label.size, seg_label.size)
        return img, cls_label,label_x,label_y

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f,self.row_anchor))

        all_idx = np.zeros((self.num_lanes,len(sample_tmp),2))
        for i,r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i,:,1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i,valid,:]
            # get all valid lane points
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
