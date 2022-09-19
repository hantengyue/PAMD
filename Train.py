import torch
import torch.optim as optim
import datetime
import os
import pickle as pk
import numpy as np
from dataset import get_valid_batch_data
from utils import get_neg_sam_item, neg_sample_compute
import torch.nn.functional as F


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


def decoder_loss_function(img_rep, de_txt, de_txt_c, de_txt_s, t):
    img = F.normalize(img_rep, dim=1)
    txt = F.normalize(de_txt, dim=1)
    txt_c = F.normalize(de_txt_c, dim=1)
    txt_s = F.normalize(de_txt_s, dim=1)
    pos_1 = torch.sum(img * txt_c, dim=1)
    pos_2 = torch.sum(img * txt, dim=1)
    neg_1 = torch.sum(img * txt_s)
    pos_1_h = torch.exp(pos_1 / t)
    pos_2_h = torch.exp(pos_2 / t)
    neg_1_h = torch.exp(neg_1 / t)
    loss_1 = -torch.mean(torch.log(pos_1_h/(pos_1_h + pos_2_h + neg_1_h) + 1e-24))
    loss_2 = -torch.mean(torch.log(pos_2_h/(pos_2_h + neg_1_h) + 1e-24))
    return loss_1 + loss_2


def Train(model, train_dataloader, l_r, weight_decay, num_epoch, save_dir, dataset, tem):
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=l_r, weight_decay=weight_decay)
    Loss_MSE = torch.nn.MSELoss(reduction='mean')
    for i in range(num_epoch):
        t1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('epoch:', i, '-----', t1)
        # 设置模型模式
        model.train()
        epoch_loss = 0.0
        for j, data in enumerate(train_dataloader):
            # print('batch', j)
            # batch_img, batch_text = data  # 已经变成tensor了
            batch_item = data
            # print(batch_img)
            t_batch_item = batch_item.cuda().long()
            # t_batch_img = batch_img.cuda()  # 查找embedding，所以做成longtensor类型
            # t_batch_text = batch_text.cuda().long()
            t_batch_item_2d = t_batch_item.unsqueeze(1)
            # t_batch_img_3d = t_batch_img.unsqueeze(1)
            # t_batch_text_3d = t_batch_text.unsqueeze(1)
            # 放进模型
            img_c_out, txt_c_out, img_s_out, txt_s_out, img_out, txt_out, de_img_out, \
            de_img_c_out, de_img_s_out, de_txt_out, de_txt_c_out, de_txt_s_out = model(t_batch_item_2d)

            img_c_out_2d = img_c_out.squeeze()
            txt_c_out_2d = txt_c_out.squeeze()
            img_s_out_2d = img_s_out.squeeze()
            txt_s_out_2d = txt_s_out.squeeze()
            img_rep_out_2d = img_out.squeeze()  # [B,4096]
            txt_rep_out_2d = txt_out.squeeze()  # [B,2700]
            de_img_out_2d = de_img_out.squeeze()  # init [B,2700]
            de_img_c_out_2d = de_img_c_out.squeeze()  # com [B,2700]
            de_img_s_out_2d = de_img_s_out.squeeze()  # spe [B,2700]
            de_txt_out_2d = de_txt_out.squeeze()  # init [B,4096]
            de_txt_c_out_2d = de_txt_c_out.squeeze()  # com [B, 4096]
            de_txt_s_out_2d = de_txt_s_out.squeeze()   # spe [B,4096]

            L_sim = Loss_MSE(img_c_out_2d, txt_c_out_2d)
            dot_prod = torch.sum(torch.mul(img_s_out_2d, txt_s_out_2d), dim=1)
            L_ort = torch.mean(dot_prod ** 2, dim=0)

            decoder_loss_T2V = decoder_loss_function(img_rep_out_2d, de_txt_out_2d, de_txt_c_out_2d, de_txt_s_out_2d, tem)
            decoder_loss_V2T = decoder_loss_function(txt_rep_out_2d, de_img_out_2d, de_img_c_out_2d, de_img_s_out_2d, tem)

            # img_init_com = Loss_MSE(de_img_out_2d, txt_rep_out_2d)-Loss_MSE(de_img_c_out_2d, txt_rep_out_2d)
            # L_init_com_img = -torch.log(torch.sigmoid(img_init_com) + 1e-24)
            # img_spe_init = Loss_MSE(de_img_s_out_2d, txt_rep_out_2d)-Loss_MSE(de_img_out_2d, txt_rep_out_2d)
            # L_spe_init_img = -torch.log(torch.sigmoid(img_spe_init) + 1e-24)
            #
            # txt_init_com = Loss_MSE(de_txt_out_2d, img_rep_out_2d) - Loss_MSE(de_txt_c_out_2d, img_rep_out_2d)
            # L_init_com_txt = -torch.log(torch.sigmoid(txt_init_com) + 1e-24)
            # txt_spe_init = Loss_MSE(de_txt_s_out_2d, img_rep_out_2d) - Loss_MSE(de_txt_out_2d, img_rep_out_2d)
            # L_spe_init_txt = -torch.log(torch.sigmoid(txt_spe_init) + 1e-24)

            # L = L_sim + L_ort + L_init_com_img + L_spe_init_img + L_init_com_txt + L_spe_init_txt
            # L = L_sim + L_ort
            L = L_sim + L_ort + decoder_loss_T2V + decoder_loss_V2T

            optimizer.zero_grad()
            epoch_loss += L.item()
            # print('batch:', j, 'batch_loss', out.item())
            L.backward()
            optimizer.step()
            # print('batch:', j)
        print('epoch', i, 'epoch_loss', epoch_loss / (j + 1))
        f1 = open(os.path.join(save_dir, dataset, 'log.txt'), 'a')
        f1.write('epoch:' + ' ' + str(i) + ' ' + 'epoch_loss' + ' ' + str(epoch_loss / (j + 1)) + '\n')
        f1.close()

        # 每隔几个epoch,就保存一下模型
        if i % 10 == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i}
            save_path = save_dir + '/' + dataset + '/' + str(i) + '.pth'
            torch.save(state, save_path)


def BPR_Train(model, bpr_train_dataloader, bpr_l_r, finetune_l_r, weight_decay, num_epoch, save_file, fine_dataset,
              item_num, batch_size_valid, save_dir, dataset, valid, test, user_item_dict, eval_neg_sam):
    # 写微调的优化器，取数据，放入模型，训练，测试结果
    pre_params = list(map(id, model.PreTrain.parameters()))
    remain_params = filter(lambda p: id(p) not in pre_params, model.parameters())
    bpr_optimizer = optim.Adam([{'params': model.PreTrain.parameters()},
                                {'params': remain_params, 'lr': bpr_l_r}], lr=finetune_l_r, weight_decay=weight_decay)
    batch_idx, starts_ends = compute_start_end(item_num, batch_size_valid)

    for i in range(num_epoch):
        t1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('epoch:', i, '-----', t1)
        # 设置模型模式
        model.train()
        bpr_epoch_loss = 0.0
        for j, data in enumerate(bpr_train_dataloader):
            # print('epoch:', j, '-----', t1)
            # print('batch', j)
            user, batch_item, batch_neg_item = data  # 已经变成tensor了
            t_user = user.cuda().long()
            t_batch_item = batch_item.cuda().long()
            t_batch_neg_item = batch_neg_item.cuda().long()
            # 放进模型
            out = model(t_user, t_batch_item, t_batch_neg_item)
            bpr_optimizer.zero_grad()
            bpr_epoch_loss += out.item()
            # print('batch:', j, 'batch_loss', out.item())
            out.backward()
            bpr_optimizer.step()
            # print('batch:', j)
        print('epoch', i, 'epoch_loss', bpr_epoch_loss / (j + 1))
        f1 = open(os.path.join(save_file, fine_dataset, 'log.txt'), 'a')
        f1.write('epoch:' + ' ' + str(i) + ' ' + 'epoch_loss' + ' ' + str(bpr_epoch_loss / (j + 1)) + '\n')
        f1.close()

        # 每隔几个epoch,就保存一下模型
        if i % 20 == 0:
            state = {'model': model.state_dict(), 'optimizer': bpr_optimizer.state_dict(), 'epoch': i}
            save_path = save_dir + '/' + dataset + '/' + str(i) + '.pth'
            torch.save(state, save_path)
            # 测试性能，valid和test

            model.eval()
            with torch.no_grad():
                val_pre = []
                val_recall = []
                val_ndcg = []

                for key, value in valid.items():
                    user_all_inter = user_item_dict[key]
                    val_neg_smp_1000_item = get_neg_sam_item(item_num, user_all_inter, eval_neg_sam, value)
                    val_u = torch.LongTensor([key]).cuda()
                    val_i = torch.from_numpy(np.array(val_neg_smp_1000_item)).cuda().long()
                    val_score, val_atten_score = model.test_performance(val_u, val_i)
                    # print(val_score)
                    val_score_arr = val_score.cpu().numpy()
                    # valid_item_list = val_neg_smp_1000_item.tolist()
                    v_valid_pre, v_valid_recall, v_valid_NDCG = neg_sample_compute(val_neg_smp_1000_item, val_score_arr, value, 10)
                    val_pre.append(v_valid_pre)
                    val_recall.append(v_valid_recall)
                    val_ndcg.append(v_valid_NDCG)

                v_pre = np.mean(np.array(val_pre))
                v_re = np.mean(np.array(val_recall))
                v_ndcg = np.mean(np.array(val_ndcg))
                print('valid_precision:', v_pre)
                print('valid_recall', v_re)
                print('valid_ndcg', v_ndcg)

                te_pre = []
                te_recall = []
                te_ndcg = []

                for key, value in test.items():
                    user_all_inter = user_item_dict[key]
                    test_neg_smp_1000_item = get_neg_sam_item(item_num, user_all_inter, eval_neg_sam, value)
                    test_u = torch.LongTensor([key]).cuda()
                    test_i = torch.from_numpy(np.array(test_neg_smp_1000_item)).cuda().long()
                    test_score, test_atten_socre = model.test_performance(test_u, test_i)
                    test_score_arr = test_score.cpu().numpy()
                    # test_item_list = test_neg_smp_1000_item.tolist()
                    t_precision, t_recall, t_valid_NDCG = neg_sample_compute(test_neg_smp_1000_item, test_score_arr,
                                                                             value, 10)
                    te_pre.append(t_precision)
                    te_recall.append(t_recall)
                    te_ndcg.append(t_valid_NDCG)

                t_pre = np.mean(np.array(te_pre))
                t_re = np.mean(np.array(te_recall))
                t_ndcg = np.mean(np.array(te_ndcg))
                print('test_precision:', t_pre)
                print('test_recall:', t_re)
                print('test_ndcg', t_ndcg)



                '''
                user_rep = model.get_user_emb().cpu().numpy()
                item_rep = np.empty((item_num, user_rep.shape[1]), dtype=np.float)
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
                    u_v = valid[p]
                    u_t = test[p]
                    # 负采样1000个商品和正样本，并得到商品list 与 score的list
                    user_all_inter = user_item_dict[p]
                    valid_item, valid_score, test_item, test_score = get_neg_sam_item_and_score(item_num,
                                                                                                user_all_inter,
                                                                                                eval_neg_sam,
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
                f1 = open(os.path.join(save_dir, dataset, 'log.txt'), 'a')
                f1.write('epoch:' + ' ' + str(i) + ' ' + 'valid_precision_20:' + ' ' + str(v_pre)
                         + 'valid_recall_20:' + ' ' + str(v_re) + ' ' +
                         'valid_ndcg_20:' + ' ' + str(v_ndcg) + '\n')
                f1.write('epoch:' + ' ' + str(i) + ' ' + 'test_precision_20:' + ' ' + str(t_pre)
                         + 'test_recall_20:' + ' ' + str(t_re) + ' ' +
                         'test_ndcg_20:' + ' ' + str(t_ndcg) + '\n')
                f1.close()
        if i > 200 and i % 40 == 0:
            state = {'model': model.state_dict(), 'optimizer': bpr_optimizer.state_dict(), 'epoch': i}
            save_path = save_dir + '/' + dataset + '/' + str(i) + '.pth'
            torch.save(state, save_path)
            # 测试性能，valid和test

            model.eval()
            with torch.no_grad():
                user_rep = model.get_user_emb().cpu().numpy()
                item_rep = np.empty((item_num, user_rep.shape[1]), dtype=np.float)
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
                    u_v = valid[p]
                    u_t = test[p]
                    # 负采样1000个商品和正样本，并得到商品list 与 score的list
                    user_all_inter = user_item_dict[p]
                    valid_item, valid_score, test_item, test_score = get_neg_sam_item_and_score(item_num,
                                                                                                user_all_inter,
                                                                                                eval_neg_sam,
                                                                                                s, u_v, u_t)
                    # valid_hr, valid_ndcg, test_hr, test_ndcg = compute_for_each_user(s, u_v, u_t)
                    v_valid_pre, v_valid_recall, v_valid_NDCG = neg_sample_compute(valid_item, valid_score, u_v, 20)
                    t_precision, t_recall, t_valid_NDCG = neg_sample_compute(test_item, test_score, u_t, 20)

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
                f1 = open(os.path.join(save_dir, dataset, 'log.txt'), 'a')
                f1.write('epoch:' + ' ' + str(i) + ' ' + 'valid_precision_20:' + ' ' + str(v_pre)
                         + 'valid_recall_20:' + ' ' + str(v_re) + ' ' +
                         'valid_ndcg_20:' + ' ' + str(v_ndcg) + '\n')
                f1.write('epoch:' + ' ' + str(i) + ' ' + 'test_precision_20:' + ' ' + str(t_pre)
                         + 'test_recall_20:' + ' ' + str(t_re) + ' ' +
                         'test_ndcg_20:' + ' ' + str(t_ndcg) + '\n')
                f1.close()
                '''