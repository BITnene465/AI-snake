# 用于设置配置文件，方便后续修改
import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class ConfigSet:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def add_item(self, item, update=True):
        for key in item.keys():
            bf = self.__dict__.get(key, False)
            if bf is False:
                self.__dict__.update({key: item[key]})
            elif update:
                self.__dict__[key] = item[key]

    def __getitem__(self, keys):
        if isinstance(keys, str):
            assert hasattr(self, keys), "Can't find the setting configs {}".format(keys)
            return self.__dict__.get(keys)
        elif isinstance(keys, list):
            new_list = []
            for each in keys:
                assert hasattr(self, each), "Can't find the setting configs {}".format(each)
                new_list.append(self.__dict__.get(each))
            return new_list
        else:
            raise ValueError("When you get config data, please input string or list")

    def __len__(self):
        return len(self.__dict__)

    def getkey(self):
        return self.__dict__.keys()


if __name__ == '__main__':
    config = ConfigSet()
    config.add_item({'keys': 3})
    print(config['key'])
