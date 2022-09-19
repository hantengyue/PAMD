import pickle
from torch.utils.data import Dataset
import numpy
import torch
from PIL import Image
from torchvision import transforms
import random

ima2tensor = transforms.Compose([transforms.Resize((100, 100)),
    transforms.ToTensor(),
    ])

def handle_img(path):
    org_ima = Image.open(path)
    # org_ima = org_ima.resize((224, 224))
    org_ima = org_ima.convert('RGB')
    ima_ten = ima2tensor(org_ima)
    return ima_ten


class TrainingDataset(Dataset):
    def __init__(self, item_list):
        self.item = item_list
        # self.item_img = fea_img_dict
        # self.item_title_index = title_dict

    def __len__(self):
        return len(self.item)

    def __getitem__(self, index):
        item = self.item[index]
        # item_img = self.item_img[item]
        # img_tensor = torch.from_numpy(item_img)
        # title_list = self.item_title_index[item]
        # title_tensor = torch.from_numpy(numpy.array(title_list)).long()
        # return img_tensor, title_tensor   # img_tensor [C, W, H] title_tensor [9]
        return item


class bpr_TrainingDataset(Dataset):
    def __init__(self, inter_list, user_item_dict, item_num, neg_num):
        self.interactions = inter_list
        self.user_item_dict = user_item_dict
        self.neg_num = neg_num
        self.all_set = set(range(0, item_num))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        interaction = self.interactions[index]
        user = interaction[0]
        item = interaction[1]
        '''
        # 负样本采样为10时这样取
        inter_item_set = set(self.user_item_dict[user])
        neg_item = random.sample((self.all_set - inter_item_set), self.neg_num)
        neg_item_tensor = torch.from_numpy(numpy.array(neg_item)).long()
        '''
        while True:
            neg_item = random.sample(self.all_set, 1)[0]  # random.sample返回的是一个列表，长度为k,这里是1
            if neg_item not in self.user_item_dict[user]:
                break
        return user, item, neg_item

        # return user, item, neg_item_tensor  # user[B], item[B], neg_item_tensor[B,L]


def get_valid_batch_data(batch_num, starts_ends):
    batch_item = starts_ends[batch_num]
    # print(batch_item_index)
    # print(item_list)
    # batch_item = item_list[batch_item_index[0]:(batch_item_index[-1]+1)]
    # 初始化tensor用来放img,和txt
    # batch_img = torch.empty(len(batch_item), 4096)
    # batch_txt = torch.empty(len(batch_item), title_len)
    # for i, item in enumerate(batch_item):
    #     img_ten = item_img_dict[item_int2str_dict[item]]
    #     img_ten = handle_img(img_path)
        # batch_img[i] = torch.from_numpy(img_ten)
        # txt = item_txt_dict[item_int2str_dict[item]]
        # batch_txt[i] = torch.from_numpy(numpy.array(txt)).long()
    return batch_item

