import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils import StandardScaler, MaxMinScaler

from tqdm import tqdm
from time import time


def latlon2xyz(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def load_dataset(
    txt_path="data_process/path.txt",
    station_path="data/small_weather2K_split60.npy",
    station_mean_std_path="data_process/mean_std_d60.npy",
    image_path="data/big_160_60_16.npy",
    image_max_min_path="data_process/max_min_d60.npy",
    val_batch_size=None,
    batch_size=32,
    label_id=21,
    n_his=12,
    n_pred=12,
    use_single=False,
    is_random=True,  # whether to use random data as satellite images
    use_time_emb=False,
    **kwargs
):

    if val_batch_size == None:
        val_batch_size = batch_size

    path_lis_txt = []
    with open(txt_path, "r") as f:
        for l in f:
            path_lis_txt.append(l[:-1])
    f.close()

    ### Station data
    print("Loading station data")
    station_data = np.load(station_path)

    ### Satellite data
    if not is_random:
        print("Loading satellite data")
        image_data = np.load(image_path)
    else:
        print("Random satellite data will be used")
        image_data = None


    ### Dataset split
    train_lis = station_data[: -8760 * 2]
    val_lis = station_data[-8760 * 2 : -8760]
    test_lis = station_data[-8760:]
    print("Dataset split: ", train_lis.shape[0], val_lis.shape[0], test_lis.shape[0])

    if image_data is not None:
        train_lis_image = image_data[: -8760 * 2]
        val_lis_image = image_data[-8760 * 2 : -8760]
        test_lis_image = image_data[-8760:]
    else:
        train_lis_image = None
        val_lis_image = None
        test_lis_image = None

    ### Scaler
    ms = np.load(station_mean_std_path)
    scaler = [StandardScaler(mean=ms[i][0], std=ms[i][1]) for i in range(ms.shape[0])]
    ms_image = np.load(image_max_min_path)
    scaler_image = [
        MaxMinScaler(max=ms_image[0][i], min=ms_image[1][i]) for i in range(ms_image.shape[1])
    ]

    ### Dataset
    print("Creating training set")
    train_set = Dataset(
        train_lis,
        train_lis_image,
        n_his,
        n_pred,
        scaler_station=scaler,
        scaler_image=scaler_image,
        use_single=use_single,
        label_id=label_id,
        is_random=is_random,
        use_time_emb=use_time_emb,
        path_lis_txt=path_lis_txt,
        start=0,
    )
    print("Creating validation set")
    validation_set = Dataset(
        val_lis,
        val_lis_image,
        n_his,
        n_pred,
        scaler_station=scaler,
        scaler_image=scaler_image,
        use_single=use_single,
        label_id=label_id,
        is_random=is_random,
        use_time_emb=use_time_emb,
        path_lis_txt=path_lis_txt,
        start=23376,
    )
    print("Creating test set")
    test_set = Dataset(
        test_lis,
        test_lis_image,
        n_his,
        n_pred,
        scaler_station=scaler,
        scaler_image=scaler_image,
        use_single=use_single,
        label_id=label_id,
        is_random=is_random,
        use_time_emb=use_time_emb,
        path_lis_txt=path_lis_txt,
        start=32136,
    )

    ### Dataloader
    data = {}
    data["train_loader"] = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    data["val_loader"] = torch.utils.data.DataLoader(
        validation_set, batch_size=val_batch_size, shuffle=False, num_workers=4
    )
    data["test_loader"] = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, num_workers=4
    )

    data["scaler"] = scaler

    return data


def data_transform(data, n_his, n_pred, override=True):
    # produce data slices for x_data and y_data
    if override:
        len_record = len(data)
        num = len_record - n_his - n_pred + 1
        x = np.zeros([num, 2], int)
        y = np.zeros([num, 2], int)

        for i in range(num):
            x[i, 0] = i
            x[i, 1] = i + n_his
            y[i, 0] = i + n_his
            y[i, 1] = i + n_his + n_pred
    else:
        len_record = len(data)
        start_list = []
        for i in range(0, len_record, 12):
            if i + n_his + n_pred <= len_record:
                start_list.append(i)
        x = np.zeros([len(start_list), 2], int)
        y = np.zeros([len(start_list), 2], int)
        for i, s in enumerate(start_list):
            x[i, 0] = s
            x[i, 1] = s + n_his
            y[i, 0] = s + n_his
            y[i, 1] = s + n_his + n_pred
    return x, y


class Dataset(Dataset):
    """IRDataset dataset."""

    def __init__(
        self,
        data_lis,
        image_data,
        n_his,
        n_pred,
        scaler_station=None,
        scaler_image=None,
        use_single=False,
        label_id=None,
        is_random=False,
        override=True,
        use_time_emb=False,
        path_lis_txt=None,
        start=0,
    ):
        self.use_time_emb = use_time_emb
        self.path_lis_txt = path_lis_txt
        self.start = start

        self.data_lis = data_lis
        self.image_data = image_data

        self.use_single = use_single
        self.label_id = label_id
        self.is_random = is_random
        if scaler_station != None:
            self.scaler_station = scaler_station
        if scaler_image != None:
            self.scaler_image = scaler_image

        self.x, self.y = data_transform(self.data_lis, n_his, n_pred, override)

        for i in range(len(scaler_station)):
            self.data_lis[..., i] = self.scaler_station[i].transform(
                self.data_lis[..., i]
            )
        if image_data is not None:
            for i in range(len(scaler_image)):
                self.image_data[..., i] = self.scaler_image[i].transform(self.image_data[..., i])

    def range2np(self, d_range, is_label, data_source="station"):
        if data_source == "station":
            res = self.data_lis[d_range[0] : d_range[1]]
            if is_label:
                temp = res[:, :, self.label_id : self.label_id + 1]
            else:
                if self.use_time_emb:
                    time_info = []
                    for i in range(d_range[0], d_range[1]):
                        n = self.path_lis_txt[self.start + i]
                        month = n[9:11]
                        day = n[11:13]
                        time = n[13:15]
                        mdt = np.array([int(month), int(day), int(time)])
                        mdt_1c = mdt[None]
                        mdt_nc = mdt_1c.repeat(res.shape[1], 0)
                        time_info.append(mdt_nc)
                    mdt_tnc = np.stack(time_info, axis=0)
                    res = np.concatenate([res, mdt_tnc], axis=-1)
                temp = res
        elif data_source == "image":
            res = self.image_data[d_range[0] : d_range[1]]
            temp = res
        return temp

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data_range = self.x[idx]
        labels_range = self.y[idx]

        if self.is_random:
            image_data = np.random.randn(12, 160, 160, 6)
        else:
            image_data = self.range2np(data_range, is_label=False, data_source="image")
        station_data = self.range2np(data_range, is_label=False, data_source="station")
        station_labels = self.range2np(
            labels_range, is_label=True, data_source="station"
        )

        return station_data, station_labels, image_data
