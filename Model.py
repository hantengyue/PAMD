import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner_hid, dropout=dropout)

    def forward(self, inputs):
        attn_output, slf_attn_weight = self.slf_attn(
            inputs, inputs, inputs)
        pffn_output = self.pos_ffn(attn_output)
        return pffn_output


class PreTrain(nn.Module):
    def __init__(self, args, intitem_fea_img, intitem_title_int, word_vec_arr):
        super(PreTrain, self).__init__()
        self.num_item = args.item_num
        self.title_len = args.title_len
        self.word_num = args.word_num
        self.word_dim = args.word_dim
        self.dropratio = args.dropratio
        self.img_dim = args.img_dim
        self.pre_emb_dim = args.pre_emb_dim
        self.title_len = args.title_len

        self.fea_img = nn.Embedding(self.num_item, self.img_dim)
        self.fea_img.weight.data.copy_(torch.from_numpy(intitem_fea_img))
        self.fea_img.weight.requires_grad = False

        self.title_index = nn.Embedding(self.num_item, self.title_len)
        self.title_index.weight.data.copy_(torch.from_numpy(intitem_title_int))
        self.title_index.weight.requires_grad = False

        self.word_vec = nn.Embedding(self.word_num, self.word_dim)
        self.word_vec.weight.data.copy_(torch.from_numpy(word_vec_arr))
        self.word_vec.weight.requires_grad = False

        self.en_img = nn.Linear(self.img_dim, self.pre_emb_dim, bias=True)
        self.en_txt = nn.Linear(self.title_len * self.word_dim, self.pre_emb_dim, bias=True)
        self.de_img = nn.Linear(self.pre_emb_dim, self.title_len * self.word_dim, bias=True)
        self.de_txt = nn.Linear(self.pre_emb_dim, self.img_dim, bias=True)
        '''
        self.en_img = nn.Sequential(
            nn.Linear(self.img_dim, int(self.img_dim * 0.5), bias=True),
            nn.ReLU(),
            nn.Linear(int(self.img_dim * 0.5), self.pre_emb_dim, bias=True),
            nn.ReLU(),
            nn.LayerNorm(self.pre_emb_dim, eps=1e-6)
        )

        self.en_txt = nn.Sequential(
            nn.Linear(self.title_len * self.word_dim, int(self.title_len * self.word_dim * 0.5), bias=True),
            nn.ReLU(),
            nn.Linear(int(self.title_len * self.word_dim * 0.5), self.pre_emb_dim, bias=True),
            nn.ReLU(),
            nn.LayerNorm(self.pre_emb_dim, eps=1e-6)
        )

        self.de_img = nn.Sequential(
            nn.Linear(self.pre_emb_dim, int(self.title_len * self.word_dim * 0.5), bias=True),
            nn.ReLU(),
            nn.Linear(int(self.title_len * self.word_dim * 0.5), (self.title_len * self.word_dim), bias=True),
            nn.ReLU(),
            # nn.LayerNorm(self.emb_dim, eps=1e-6)
        )

        self.de_txt = nn.Sequential(
            nn.Linear(self.pre_emb_dim, int(self.img_dim * 0.5), bias=True),
            nn.ReLU(),
            nn.Linear(int(self.img_dim * 0.5), self.img_dim, bias=True),
            nn.ReLU(),
            # nn.LayerNorm(self.emb_dim, eps=1e-6)
        )
        '''
        self.W_1 = nn.Parameter(torch.FloatTensor(self.pre_emb_dim, self.pre_emb_dim))  # 图像
        torch.nn.init.kaiming_uniform_(self.W_1)
        # 转置就用self.V.T
        self.W_2 = nn.Parameter(torch.FloatTensor(self.pre_emb_dim, self.pre_emb_dim))
        torch.nn.init.kaiming_uniform_(self.W_2)

        # self.de_en_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, batch_item):  # [B,L], [B,L]
        img_t_rep = self.fea_img(batch_item)  # [B,L,4096]
        text_title_look = self.title_index(batch_item)  # [B,L,9]
        img_rep = self.en_img(img_t_rep)  # [B,L,4096]-->[B,L,50]
        txt_emb = self.word_vec(text_title_look.long())  # [B,L,9] --> [B,L,9,300]
        change_txt_emb = txt_emb.reshape((txt_emb.shape[0], txt_emb.shape[1], txt_emb.shape[2] * txt_emb.shape[3]))   # [B,L,2700]
        txt_rep = self.en_txt(change_txt_emb)  # [B,L,50]

        txt_c = torch.matmul(txt_rep, self.W_1)   # [B,1,50]*[50,50] -->[B,1,50]
        # text_en_de = torch.matmul(text_en, self.W_2)
        img_c = torch.matmul(img_rep, self.W_2)   # [B,1,50]*[50,50] -->[B,1,50]
        # img_en_de = torch.matmul(img_en, self.W_1.T)
        img_s = img_rep - img_c   # [B,1,50]
        txt_s = txt_rep - txt_c   # [B,1,50]

        # 解码结果
        de_img = self.de_img(img_rep)
        de_img_c = self.de_img(img_c)
        de_img_s = self.de_img(img_s)

        de_txt = self.de_txt(txt_rep)
        de_txt_c = self.de_txt(txt_c)
        de_txt_s = self.de_txt(txt_s)

        return img_c, txt_c, img_s, txt_s, img_t_rep, change_txt_emb, de_img, de_img_c, de_img_s, de_txt, de_txt_c, de_txt_s


class Net(nn.Module):
    def __init__(self, args, intitem_fea_img, intitem_title_int, word_vec_arr):
        super(Net, self).__init__()
        self.item_num = args.item_num
        self.user_num = args.user_num

        # self.text_dim = args.text_dim
        self.pre_dim = args.pre_emb_dim
        self.emb_dim = args.emb_dim

        self.dropratio = args.dropratio
        self.user_emb = nn.Embedding(self.user_num, self.emb_dim)

        self.item_emb = nn.Embedding(self.item_num, self.emb_dim)
        # self.textCNN = TextCNN(self.word_num, self.word_dim, self.text_dim, self.kernel_num, self.kernel_sizes, self.dropratio)
        # self.CNN = ImgCNN(self.img_dim)

        self.PreTrain = PreTrain(args, intitem_fea_img, intitem_title_int, word_vec_arr)

        self.multihead = EncoderLayer(self.emb_dim, self.emb_dim, 1, self.emb_dim, self.emb_dim)

        # self.mlp_layer = nn.Linear(4 * self.emb_dim, self.emb_dim, bias=True)

        # self.ima_linear = nn.Linear(self.pre_dim, self.emb_dim, bias=True)
        # self.com_linear = nn.Linear(self.pre_dim, self.emb_dim, bias=True)
        # self.txt_linear = nn.Linear(self.pre_dim, self.emb_dim, bias=True)
        # self.linear = nn.Linear(self.pre_dim)
        # self.cat_linear = nn.Linear(self.text_dim+self.img_dim+self.text_dim, self.emb_dim, bias=True)
        # self.add_leaner = nn.Linear(self.img_dim, self.emb_dim, bias=True)

    def forward(self, user, item, neg_item):
        # user[B], item[B], neg_item[B,L]
        user_rep = self.user_emb(user)   # [B,d]
        mix_img_txt, _ = self.get_att_item_rep(user, item)
        mix_neg_img_txt, _ = self.get_att_item_rep(user, neg_item)
        '''
        mix_img_txt = self.get_cat_item_emb(item)  # [B, 3d]
        # mix_img_txt = self.get_add_item_emb(item)   # [B, d]
        # 这是负采样10的时候，采取的处理方式，由于这样没办法直接用multihead,所以改用负采样1个
        # mix_neg_img_txt = self.get_cat_neg_item_emb(neg_item)  # [B, 10, 3d]
        mix_neg_img_txt = self.get_cat_item_emb(neg_item)  # 这里的neg_item形状是[B]
        '''
        # if train is True:
            # 计算BPR_loss
        pos_score = torch.sum(user_rep * mix_img_txt, dim=1)  # [B]
        # 负采样个数多的时候才这样算
        # user_rep_3d = user_rep.unsqueeze(1)
        # neg_score_3d = torch.matmul(user_rep_3d, mix_neg_img_txt.permute(0, 2, 1))
        # neg_score = neg_score_3d.squeeze()  # [B,L]
        # 负采样个数为1
        neg_score = torch.sum(user_rep * mix_neg_img_txt, dim=1)
        # cha_score = pos_score - neg_score
        positive_loss = -torch.sum(torch.log(torch.sigmoid(pos_score) + 1e-24))
        negative_loss = -torch.sum(torch.log(1 - torch.sigmoid(neg_score) + 1e-24))
        loss = positive_loss + negative_loss
        # L = -torch.sum(torch.log(torch.sigmoid(cha_score) + 1e-24), dim=0)  # [B,L-1]
        return loss

    def get_user_emb(self):
        return self.user_emb.weight.data

    def get_cat_item_emb(self, item):  # [B]

        item_emb_rep = self.item_emb(item)  # [B,d]
        # 都添加第二维度，直接进self.PreTrain
        # img_3d = img.unsqueeze(1)
        # txt_3d = txt.unsqueeze(1)
        item_2d = item.unsqueeze(1)  # [B,1,d]

        img_c_3d, txt_c_3d, img_s_3d, txt_s_3d, _, _, _, _, _, _, _, _ = self.PreTrain(item_2d)
        img_c_2d = img_c_3d.squeeze()  # [B,d]
        txt_c_2d = txt_c_3d.squeeze()
        img_s_2d = img_s_3d.squeeze()
        txt_s_2d = txt_s_3d.squeeze()
        common = (img_c_2d + txt_c_2d) * 0.5
        # 利用得到的这4个表达，经过multihead
        cat_rep = torch.cat((item_emb_rep, img_s_2d, txt_s_2d, common), dim=-1)  # [B,4*d]
        cat_rep_3d = cat_rep.reshape(item_emb_rep.shape[0], 4, item_emb_rep.shape[1])  # [B,4,d]
        # multihead的输入已经做好
        encoded_item_rep = self.multihead(cat_rep_3d)  # [B,4,d]
        final_item_rep = encoded_item_rep.reshape(encoded_item_rep.shape[0], encoded_item_rep.shape[1] * encoded_item_rep.shape[2])
        # f_final_item_rep = self.mlp_layer(final_item_rep)
        # com_lin_en = self.com_linear(commom)
        # img_s_2d_lin_en = self.ima_linear(img_s_2d)
        # txt_s_2d_lin_en = self.txt_linear(txt_s_2d)
        # cat_img_txt = torch.cat((img_s_2d, common, txt_s_2d, item_emb_rep), dim=-1)
        # mix_cat_img_txt = self.cat_linear(cat_img_txt)  # relu函数用的不妥当，导致item的表达中有很多0
        return final_item_rep

    def get_att_item_rep(self, user, item):  #[B],[B]
        user_emb_rep = self.user_emb(user)   # [B,d]
        item_emb_rep = self.item_emb(item)  # [B,d]
        item_2d = item.unsqueeze(1)  # [B,1,d]

        img_c_3d, txt_c_3d, img_s_3d, txt_s_3d, _, _, _, _, _, _, _, _ = self.PreTrain(item_2d)
        img_c_2d = img_c_3d.squeeze()  # [B,d]
        txt_c_2d = txt_c_3d.squeeze()
        img_s_2d = img_s_3d.squeeze()
        txt_s_2d = txt_s_3d.squeeze()

        a_1 = torch.sum(user_emb_rep * img_c_2d, dim=1)  # [B]
        a_2 = torch.sum(user_emb_rep * txt_c_2d, dim=1)
        a_3 = torch.sum(user_emb_rep * img_s_2d, dim=1)
        a_4 = torch.sum(user_emb_rep * txt_s_2d, dim=1)

        b_1 = a_1.unsqueeze(1)
        b_2 = a_2.unsqueeze(1)
        b_3 = a_3.unsqueeze(1)
        b_4 = a_4.unsqueeze(1)

        att = torch.cat((b_1, b_2, b_3, b_4), dim=1)
        softed_att = F.softmax(att, dim=1)   # [B,4]

        c_1 = softed_att[:, 0].unsqueeze(1).repeat(1, img_c_2d.shape[1])
        c_2 = softed_att[:, 1].unsqueeze(1).repeat(1, img_c_2d.shape[1])
        c_3 = softed_att[:, 2].unsqueeze(1).repeat(1, img_c_2d.shape[1])
        c_4 = softed_att[:, 3].unsqueeze(1).repeat(1, img_c_2d.shape[1])

        e_1 = c_1 * img_c_2d
        e_2 = c_2 * txt_c_2d
        e_3 = c_3 * img_s_2d
        e_4 = c_4 * txt_s_2d

        final_item_rep = item_emb_rep + e_1 + e_2 + e_3 + e_4  # [B,d]
        return final_item_rep, softed_att

    def test_performance(self, user, item):  # [1], [1000]
        b_u = user.repeat(item.shape[0])  # [1000]
        b_item_rep, att_score = self.get_att_item_rep(b_u, item)
        user_emb_rep = self.user_emb(user)
        score = torch.sum(user_emb_rep * b_item_rep, dim=1)  # [B]
        return score, att_score


    def get_cat_neg_item_emb(self, item):  # [B,10, 4096], [B,10,9]
        item_emb_rep = self.item_emb(item)  # [B,10,d]
        img_c_3d, txt_c_3d, img_s_3d, txt_s_3d, _, _, _, _, _, _, _, _ = self.PreTrain(item)  # [B,10,d]
        common = (img_c_3d + txt_c_3d) * 0.5
        # com_lin = self.com_linear(commom)
        # img_s_3d_lin = self.ima_linear(img_s_3d)
        # txt_s_3d_lin = self.txt_linear(txt_c_3d)
        cat_img_txt = torch.cat((img_s_3d, common, txt_s_3d, item_emb_rep), dim=-1)  # [B,10,4d]
        # mix_cat_img_txt = self.cat_linear(cat_img_txt)  # relu函数用的不妥当，导致item的表达中有很多0
        return cat_img_txt

    def get_add_item_emb(self, item):
        img_rep = self.image_fea(item)
        img_en_rep = self.image_en_fea(item)
        txt_rep = self.txt_fea(item)
        txt_en_rep = self.txt_en_fea(item)
        commom = (img_en_rep + txt_en_rep) * 0.5
        add_img_txt = img_rep + commom + txt_rep
        # mix_add_img_txt = self.cat_linear(add_img_txt)
        return add_img_txt