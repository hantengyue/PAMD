import numpy as np
import random
import heapq

def compute_hr_ndcg(score, valid, test):
    val_hr = []
    val_ndcg = []
    te_hr = []
    te_ndcg = []
    for i in range(score.shape[0]):
        user_score = score[i]
        valid_item = valid[i]
        test_item = test[i]
        valid_hr, valid_ndcg, test_hr, test_ndcg = compute_for_each_user(user_score, valid_item, test_item)
        val_hr = val_hr + valid_hr
        val_ndcg.append(valid_ndcg)
        te_hr = te_hr + test_hr
        te_ndcg.append(test_ndcg)
    pre_val_hr = np.array(val_hr).flatten()
    pre_te_hr = np.array(te_hr).flatten()
    v_h = np.mean(pre_val_hr)
    t_h = np.mean(pre_te_hr)
    v_n = np.mean(np.array(val_ndcg))
    t_n = np.mean(np.array(te_ndcg))
    return v_h, v_n, t_h, t_n


def compute_for_each_user(user_sco, valid_list, test_list):
    sorted_score = np.sort(user_sco)
    sorted_index = np.argsort(user_sco)
    top_20_daoxu = sorted_index[-20:]
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


def get_neg_sam_item(item_num, user_all_inter, eval_neg_sam, u_valid):
    all_item_set = set(range(0, item_num))
    inter_item_set = set(user_all_inter)
    neg_item = random.sample((all_item_set - inter_item_set), eval_neg_sam-len(u_valid))  # 是list类型
    valid_all = neg_item + u_valid
    # test_all = neg_item + u_test
    # s 是array类型，可以直接把list当做索引
    # valid_item_score = s[valid_all]  # 是array类型
    # test_item_score = s[test_all]  # 是array类型
    return valid_all


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
    if len(u_v) != 0:
        recall = len(common_item) / len(u_v)
    else:
        recall = 0
    return precision, recall, ndcg_value


def compute_final_value(val_com_len, val_k, val_tru, val_ndcg):
    all_hit = np.array(val_com_len).sum()
    all_k = np.array(val_k).sum()
    all_tru = np.array(val_tru).sum()
    precision = all_hit / all_k
    recall = all_hit / all_tru
    ndcg = np.array(val_ndcg).mean()
    return precision, recall, ndcg


def load_data(path):
    item_list = []
    with open(path, 'r') as f:
        for line in f:
            u, i = line.strip().split('|')
            if i not in item_list:
                item_list.append(i)
    return item_list, len(item_list)


def load_img_txt(it_fea, item_num, it_size, aliases_dict):
    # print(name + ' features ...')               # name = 'Image' or 'Text'
    fea_it = np.random.rand(item_num, it_size)  # 多出来一个，存放用于补齐用户购买序列的/实际不存在的item
    for key, value in it_fea.items():
        if key in aliases_dict.keys():
            fea_it[aliases_dict[key]] = value
    return fea_it