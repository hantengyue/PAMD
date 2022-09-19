# 加载模型，输入item,得到每个item的四种表达，并且保存下来
import torch
import numpy as np
from dataset import get_valid_batch_data


def compute_start_end(item_num, batch_size_valid):
    rest = (item_num % batch_size_valid) > 0  # 能整除：rest=0。不能整除：rest=1，则多出来一个小的batch
    n_batches = np.minimum(item_num // batch_size_valid + rest, item_num)  # 计算出batch的数目
    batch_idxs = np.arange(n_batches, dtype=np.int32)
    starts_ends = []
    for bidx in batch_idxs:
        start = bidx * batch_size_valid
        end = np.minimum(start + batch_size_valid, item_num)  # 限制标号索引不能超过user_num
        start_end = np.arange(start, end, dtype=np.int32)
        starts_ends.append(start_end)
    return batch_idxs, starts_ends  # 返回的是batch的索引，和每个batch 内的索引


def get_rep(model, new_item_list, save_file, dataset, th, batch_size_valid, item_img_dict, item_txt_dict, title_len):
    load_path = save_file + '/' + dataset + '/' + str(th) + '.pth'
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    print('加载 epoch {} 成功！'.format(start_epoch))
    num_item = len(new_item_list)
    batch_idx, starts_ends = compute_start_end(num_item, batch_size_valid)
    model.eval()
    item_img = {}
    item_img_en = {}
    item_txt = {}
    item_txt_en = {}
    with torch.no_grad():
        # 在new_item_list中做batch的数据,送入模型，得到相应数据，并保存成字典，值为512维的array
        for batch_num in batch_idx:
            valid_batch_item, valid_batch_item_img, valid_batch_item_txt = get_valid_batch_data(batch_num,
                                                                                                starts_ends,
                                                                                                new_item_list,
                                                                                                item_img_dict,
                                                                                                item_txt_dict,
                                                                                                title_len)
            t_valid_batch_item_img = valid_batch_item_img.cuda()
            t_valid_batch_item_txt = valid_batch_item_txt.cuda().long()
            img, img_en, txt, txt_en = model(t_valid_batch_item_img, t_valid_batch_item_txt, train=False)
            # 把tensor拿出来，成为numpy,用字典保存
            img_arr = img.cpu().numpy()
            img_en_arr = img_en.cpu().numpy()
            txt_arr = txt.cpu().numpy()
            txt_en_arr = txt_en.cpu().numpy()
            for i, item in enumerate(valid_batch_item):
                item_img[item] = img_arr[i]
                item_img_en[item] = img_en_arr[i]
                item_txt[item] = txt_arr[i]
                item_txt_en[item] = txt_en_arr[i]
    return item_img, item_img_en, item_txt, item_txt_en

