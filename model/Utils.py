import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DataInput(object):
    def __init__(self, data_dir: str, data_split_ratio: tuple):
        self.data_dir = data_dir
        self.data_split_ratio = data_split_ratio

    def load_data(self):
        PATH = self.data_dir + '/jp20200401_20210921.npy'
        data = np.load(PATH, allow_pickle='TRUE').item()

        data_od = data['od']
        data_node_inf = np.log(data['node'][...,[0]]+1.0)
        data_node_other = data['node'][...,[1,2,3]]
        data_node = np.concatenate((data_node_inf,data_node_other),axis=-1)
        data_SIR = data['SIR']
        data_y = data['node'][...,[0]]
        commute = np.load(self.data_dir + '/commute_jp.npy')

        # return a dict
        dataset = dict()
        dataset['od'] = data_od
        dataset['node'] = data_node
        dataset['SIR'] = data_SIR
        dataset['y'] = data_y
        dataset['commute'] = commute
        return dataset


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class ODDataset(Dataset):
    def __init__(self, inputs: dict, output: torch.Tensor, mode: str, mode_len: dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_od'][item], self.inputs['x_node'][item], self.inputs['x_SIR'][item], self.output[item]

    def prepare_xy(self, inputs: dict, output: torch.Tensor):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:  # test
            start_idx = self.mode_len['train'] + self.mode_len['validate']

        x = dict()
        x['x_od'] = inputs['x_od'][start_idx: (start_idx + self.mode_len[self.mode])]
        x['x_SIR'] = inputs['x_SIR'][start_idx: (start_idx + self.mode_len[self.mode])]
        x['x_node'] = inputs['x_node'][start_idx: (start_idx + self.mode_len[self.mode])]

        y = output[start_idx: start_idx + self.mode_len[self.mode]]
        return x, y

class DataGenerator(object):
    def __init__(self, obs_len: int, pred_len, data_split_ratio: tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len: int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0] / sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1] / sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        return mode_len

    def get_data_loader(self, data: dict, params: dict):
        x_od, x_node, x_SIR, y = self.get_feats(data)
        x_od = np.asarray(x_od)
        x_node = np.asarray(x_node)
        x_SIR = np.asarray(x_SIR)
        y = np.asarray(y)

        mode_len = self.split2len(data_len=y.shape[0])

        # SCALER
        max_od = x_od[:mode_len['train'], ...,0].max()
        x_od = x_od / max_od

        for i in range(x_node.shape[-1]):
                scaler = StandardScaler(mean=x_node[:mode_len['train'],..., i].mean(),
                                        std=x_node[:mode_len['train'],..., i].std())
                x_node[...,i] = scaler.transform(x_node[...,i])

        feat_dict = dict()
        feat_dict['x_od'] = torch.from_numpy(x_od).float().to(params['GPU'])
        feat_dict['x_node'] = torch.from_numpy(x_node).float().to(params['GPU'])
        feat_dict['x_SIR'] = torch.from_numpy(x_SIR).float().to(params['GPU'])
        y = torch.from_numpy(y).float().to(params['GPU'])

        print('Data split:', mode_len)

        data_loader = dict()  # data_loader for [train, validate, test]
        data_loader['max_od'] = max_od
        for mode in ['train', 'validate', 'test']:
            dataset = ODDataset(inputs=feat_dict, output=y, mode=mode, mode_len=mode_len)
            print('Data loader', '|', mode, '|', 'input node features:', dataset.inputs['x_node'].shape, '|'
                  'output:', dataset.output.shape)
            if mode == 'train':
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)
            else:
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
        return data_loader

    def get_feats(self, data: dict):
        x_od, x_node, x_SIR, y = [], [], [], []
        for i in range(self.obs_len, data['od'].shape[0] - self.pred_len + 1):
            x_od.append(data['od'][i - self.obs_len: i])
            x_node.append(data['node'][i - self.obs_len: i])
            x_SIR.append(data['SIR'][i - self.obs_len: i])
            y.append(data['y'][i: i + self.pred_len])
        return x_od, x_node, x_SIR, y

