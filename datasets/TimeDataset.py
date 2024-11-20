import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, net, mode='train', config = None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode
        self.net = net

        x_data = raw_data[:-1]#features
        labels = raw_data[-1]#labels


        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        if self.net is None:
            self.x, self.y, self.labels = self.process(data, labels)
        else:
            self.x, self.py, self.ny, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:

            ft = data[:, i-slide_win:i]
            tar = data[:, i]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])
        if len(x_arr) == 0:
            return 0,0,0
        else:
            x = torch.stack(x_arr).contiguous()
            y = torch.stack(y_arr).contiguous()

            labels = torch.Tensor(labels_arr).contiguous()
            
            if self.net is None:
                return x, y, labels
            else:
                py = y
                ny = self.net
                return x,py,ny,labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        if self.net is None:
            y = self.y[idx].double()
            return feature, y, label, edge_index
        else:
            py = self.py[idx].double()
            ny = self.ny[idx].double()
            return feature, py, ny, label, edge_index






