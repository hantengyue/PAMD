# MGNN  pytorch 实现
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import load_img_txt
from dataset import TrainingDataset, bpr_TrainingDataset
import pickle as pk
from Model import Net, PreTrain
from Train import Train, BPR_Train
from get_rep import get_rep
import os
import heapq
import random


def get_neg_sam_item_and_score(item_num, user_all_inter, eval_neg_sam, s, u_valid, u_test):
    all_item_set = set(range(0, item_num))
    inter_item_set = set(user_all_inter)
    neg_item = random.sample((all_item_set - inter_item_set), eval_neg_sam)  # 是list类型
    valid_all = neg_item + u_valid
    test_all = neg_item + u_test
    # s 是array类型，可以直接把list当做索引
    valid_item_score = s[valid_all]  # 是array类型
    test_item_score = s[test_all]  # 是array类型
    return valid_all, valid_item_score, test_all, test_item_score


def neg_sample_compute(valid_item, valid_score, u_v, k):
    # valid_item是list, valid_score是array, u_v是list, k是top-k的数值
    index = heapq.nlargest(k, range(len(valid_score)), valid_score.take)
    # print(index)
    valid_item_arr = np.array(valid_item)[index]
    top_k_item_list = valid_item_arr.tolist()
    # 现在得到了top-k的商品，和ground_truth的商品，计算precision，recall，ndcg
    ndcg_value = get_ndcg(top_k_item_list, u_v)
    common_item = list(set(top_k_item_list).intersection(set(u_v)))
    precision = len(common_item) / k
    recall = len(common_item) / len(u_v)
    return precision, recall, ndcg_value

def compute_final_value(val_com_len, val_k, val_tru, val_ndcg):
    all_hit = np.array(val_com_len).sum()
    all_k = np.array(val_k).sum()
    all_tru = np.array(val_tru).sum()
    precision = all_hit / all_k
    recall = all_hit / all_tru
    ndcg = np.array(val_ndcg).mean()
    return precision, recall, ndcg


def compute_for_each_user(user_sco, valid_list, test_list):
    sorted_score = np.sort(user_sco)
    sorted_index = np.argsort(user_sco)
    top_20_daoxu = sorted_index[-100:]
    top_20 = list(reversed(top_20_daoxu))
    valid_hr = get_hit_ratio(top_20, valid_list)
    valid_ndcg = get_ndcg(top_20, valid_list)
    test_hr = get_hit_ratio(top_20, test_list)
    test_ndcg = get_ndcg(top_20, test_list)
    return valid_hr, valid_ndcg, test_hr, test_ndcg


def get_hit_ratio(rank_list, ground_truth):
    hr_list = []
    for item in ground_truth:
        if item in rank_list:
            hr_list.append(1.0)
        else:
            hr_list.append(0.0)
    return hr_list


def get_ndcg(rank_list, ground_truth):
    relevance = np.ones_like(ground_truth)
    it2rel = {it: r for it, r in zip(ground_truth, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def getDCG(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


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


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='F:/data-mm-bpr/food/food_item_list', help='Dataset path')
    parser.add_argument('--img_data_path',
                        default='F:/dataset/food/item_vgg16_feature_dict',
                        help='Dataset path')
    parser.add_argument('--txt_data_path', default='F:/data-mm-bpr/food/item_title_int_dict', help='Dataset path')
    parser.add_argument('--title_len', type=int, default=18, help='num of word.')
    parser.add_argument('--word_vec_arr', default='F:/data-mm-bpr/food/word_int_vec_arr', help='Dataset path')
    parser.add_argument('--inter_data', default='F:/data-mm-bpr/food/', help='Filename')

    parser.add_argument('--save_file', default='F:/mm-data/', help='Filename')
    parser.add_argument('--pre_dataset', default='food/v5/pretrain/', help='Filename')
    parser.add_argument('--fine_dataset', default='food/v5/finetune/', help='Filename')
    parser.add_argument('--pre_train', default=False, help='train or not')
    parser.add_argument('--th', type=int, default=90, help='load which model.')
    parser.add_argument('--batch_valid', type=int, default=216, help='Batch size.')

    parser.add_argument('--user_num', type=int, default=149672, help='num of word.')
    parser.add_argument('--item_num', type=int, default=39213, help='num of word.')

    # textCNN 中参数
    # parser.add_argument('--word_num', type=int, default=10892, help='num of word.')
    parser.add_argument('--word_num', type=int, default=103367, help='num of word.')
    parser.add_argument('--word_dim', type=int, default=300, help='emb dim of word.')
    parser.add_argument('--img_dim', type=int, default=4096, help='num of word.')
    parser.add_argument('--pre_emb_dim', type=int, default=30, help='num of word.')
    parser.add_argument('--emb_dim', type=int, default=30, help='num of word.')

    parser.add_argument('--dropratio', type=float, default=0.5, help='text CNN drop ratio.')

    parser.add_argument('--pre_l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Learning rate.')
    parser.add_argument('--bpr_l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--finetune_l_r', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--neg_sam', type=int, default=10, help='Batch size.')

    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--pre_num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')

    args = parser.parse_args()
    # device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # 读取所有的商品，并做字典，反字典(取图片和文本的时候会用到)
    item_list = pk.load(open(args.data_path, 'rb'))
    print('交互的商品个数：', len(item_list))
    fea_img_dict = pk.load(open(args.img_data_path, 'rb'))
    word_vec_arr = pk.load(open(args.word_vec_arr, 'rb'))
    # print(word_vec_arr)
    title_dict = pk.load(open(args.txt_data_path, 'rb'))

    item_str2int_dict = pk.load(open(args.inter_data + 'item_dict', 'rb'))
    item_int2str_dict = {}
    for key, value in item_str2int_dict.items():
        item_int2str_dict[value] = key

    # 做int item的image feature array和title array,传入模型，直接能查找
    intitem_fea_img = load_img_txt(fea_img_dict, args.item_num, args.img_dim, item_str2int_dict)
    intitem_title_int = load_img_txt(title_dict, args.item_num, args.title_len, item_str2int_dict)

    new_item_list = []
    for item in item_list:
        if item in fea_img_dict.keys() and item in title_dict.keys():
            new_item_list.append(item_str2int_dict[item])
    print('参与训练的商品个数：', len(new_item_list))
    # 用title title_item_dict
    # train_dataset = TrainingDataset(new_item_list)
    # train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

    # pre_train = PreTrain(args, intitem_fea_img, intitem_title_int, word_vec_arr)
    # pre_train = pre_train.cuda()
    # 进行训练
    # if args.pre_train is True:
    #     Train(pre_train, train_dataloader, args.pre_l_r, args.weight_decay, args.pre_num_epoch, args.save_file, args.pre_dataset)
    # else:
        # 下游任务的dataloader
    user_item_dict = pk.load(open(args.inter_data + 'user_item_dict', 'rb'))
    train_list = pk.load(open(args.inter_data + 'train_interactions', 'rb'))
    test_dict = pk.load(open(args.inter_data + 'test_dict', 'rb'))
    valid_dict = pk.load(open(args.inter_data + 'valid_dict', 'rb'))

    # bpr_train_dataset = bpr_TrainingDataset(train_list, user_item_dict, args.item_num, args.neg_sam)
    # bpr_train_dataloader = DataLoader(bpr_train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    # 加载预训练模型
    model = Net(args, intitem_fea_img, intitem_title_int, word_vec_arr)
    model = model.cuda()

    load_path = 'F:/mm-data/food/v5/finetune/dim-30/12.pth'
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    print('加载 预训练模型 epoch {} 成功！'.format(start_epoch))
    batch_idx, starts_ends = compute_start_end(args.item_num, args.batch_valid)
    model.eval()
    with torch.no_grad():
        user_rep = model.get_user_emb().cpu().numpy()
        item_rep = np.empty((args.item_num, user_rep.shape[1]), dtype=np.float)
        for batch_num in batch_idx:
            valid_batch_item = get_valid_batch_data(batch_num, starts_ends)
            t_valid_batch_item = torch.from_numpy(valid_batch_item).cuda().long()
            # t_valid_batch_img = valid_batch_img.cuda()
            # t_valid_batch_txt = valid_batch_txt.cuda().long()
            t_batch_item_rep = model.get_cat_item_emb(t_valid_batch_item)
            batch_item_rep = t_batch_item_rep.cpu().numpy()
            for q, item_id in enumerate(valid_batch_item):
                item_rep[item_id] = batch_item_rep[q]
        val_pre = []
        val_recall = []
        val_ndcg = []
        te_pre = []
        te_recall = []
        te_ndcg = []
        for p in range(user_rep.shape[0]):
            s = np.dot(user_rep[p], item_rep.T)  # 得到了用户和所有商品的score,
            u_v = valid_dict[p]
            u_t = test_dict[p]
            # 负采样1000个商品和正样本，并得到商品list 与 score的list
            user_all_inter = user_item_dict[p]
            valid_item, valid_score, test_item, test_score = get_neg_sam_item_and_score(args.item_num,
                                                                                        user_all_inter,
                                                                                        1000,
                                                                                        s, u_v, u_t)
            # valid_hr, valid_ndcg, test_hr, test_ndcg = compute_for_each_user(s, u_v, u_t)
            v_valid_pre, v_valid_recall, v_valid_NDCG = neg_sample_compute(valid_item, valid_score, u_v, 10)
            t_precision, t_recall, t_valid_NDCG = neg_sample_compute(test_item, test_score, u_t, 10)

            val_pre.append(v_valid_pre)
            val_recall.append(v_valid_recall)
            val_ndcg.append(v_valid_NDCG)

            te_pre.append(t_precision)
            te_recall.append(t_recall)
            te_ndcg.append(t_valid_NDCG)

        v_pre = np.mean(np.array(val_pre))
        v_re = np.mean(np.array(val_recall))
        v_ndcg = np.mean(np.array(val_ndcg))
        t_pre = np.mean(np.array(te_pre))
        t_re = np.mean(np.array(te_recall))
        t_ndcg = np.mean(np.array(te_ndcg))
        print('valid_precision:', v_pre)
        print('valid_recall', v_re)
        print('valid_ndcg', v_ndcg)
        print('test_precision:', t_pre)
        print('test_recall:', t_re)
        print('test_ndcg', t_ndcg)
        f1 = open(os.path.join(args.save_file, args.fine_dataset, 'log.txt'), 'a')
        f1.write('epoch:' + ' ' + 'valid_precision_20:' + ' ' + str(v_pre)
                 + 'valid_recall_20:' + ' ' + str(v_re) + ' ' +
                 'valid_ndcg_20:' + ' ' + str(v_ndcg) + '\n')
        f1.write('epoch:' + ' ' + 'test_precision_20:' + ' ' + str(t_pre)
                 + 'test_recall_20:' + ' ' + str(t_re) + ' ' +
                 'test_ndcg_20:' + ' ' + str(t_ndcg) + '\n')
        f1.close()
if __name__ == '__main__':
    main()
