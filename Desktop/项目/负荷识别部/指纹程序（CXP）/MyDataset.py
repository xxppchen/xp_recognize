from torch.utils.data import Dataset
from enum import Enum
import os
import pandas as pd
import json
import numpy as np
import copy

__all__ = ["LabelType", "MyDataset"]


class LabelType(Enum):
    Type = "type"
    Location = "location"
    Load = "load"
    # Status = "status"
    Is_heat = "is_heat"
    Is_cold = "is_cool"
    Is_rotate = "is_rotate"
    Is_light = "is_light"


def clean_meta(ist):
    """remove '' elements in Meta Data """
    ist_meta = copy.deepcopy(ist)
    ist_temp = copy.deepcopy(ist_meta['appliance'])
    for k, v in ist_temp.items():
        if len(v) == 0:
            del ist_meta['appliance'][k]
    return ist_meta


def parse_meta(meta):
    """parse meta data for easy access"""
    ms = {}
    for k, v in meta.items():
        ms[k] = clean_meta(v)  # 删除空信息
    return ms


def get_dic_value_by_k(dict_, k):
    if k.value in dict_:
        return dict_[k.value]
    else:
        for v in dict_.values():
            if isinstance(v, dict):
                value = get_dic_value_by_k(v, k)
                if value is not None:
                    return value
        return None


def get_label_list(metas, label_type):
    encode_list = {}
    labels = [get_dic_value_by_k(x, label_type) for x in metas.values()]
    unq_label = list(set(labels))
    unq_label.sort()
    for i, label in enumerate(unq_label):
        encode_list[label] = i
    return encode_list


class MyDataset(Dataset):
    def __init__(self, meta_path, csv_path, label_type, used_feas=None):
        super(MyDataset, self).__init__()
        self.meta_path = meta_path
        self.csv_path = csv_path
        with open(meta_path) as data_file:
            meta = json.load(data_file)
        self.Metas = parse_meta(meta)
        self.label_type = label_type
        # 标签编号对应列表
        self.encode_list = get_label_list(self.Metas, label_type)
        feas = pd.read_csv(csv_path, header=None, nrows=1).values
        if used_feas is None:
            self.csv_data = pd.read_csv(csv_path, sep=',')
        else:
            used_feas = list(used_feas)
            used_feas = ['id'] + used_feas
            self.csv_data = pd.read_csv(csv_path, sep=',', usecols=used_feas)

    def __getitem__(self, index):
        # 获取该数据的标签编码
        app_item_id = int(self.csv_data["id"][index])
        app_item = self.Metas[str(app_item_id + 1)]
        label_name = get_dic_value_by_k(app_item, self.label_type)
        label = self.encode_list[label_name]
        item_data = self.csv_data.iloc[index, 1:]
        # 进行数据预处理在此处进行
        return item_data, label

    def __len__(self):
        return len(self.csv_data)
