# MGNN  pytorch 实现
import argparse
import torch
from torch.utils.data import DataLoader
from utils import load_img_txt
from dataset import TrainingDataset, bpr_TrainingDataset
import pickle as pk
from Model import Net, PreTrain
from Train import Train, BPR_Train
from get_rep import get_rep


def main():
    parser = argparse.ArgumentParser()
    # 数据集Clothing参数

    parser.add_argument('--data_path', default='F:/data-mm-bpr/Clothing/Clothing_item_list', help='Dataset path')
    parser.add_argument('--img_data_path', default='F:/amazon/2014_Clothing_Shoes_and_Jewelry_5/item_vgg16_feature_dict', help='Dataset path')
    parser.add_argument('--txt_data_path', default='F:/data-mm-bpr/Clothing/item_title_int_dict', help='Dataset path')
    parser.add_argument('--title_len', type=int, default=9, help='num of word.')
    parser.add_argument('--word_vec_arr', default='F:/data-mm-bpr/Clothing/word_int_vec_arr', help='Dataset path')
    parser.add_argument('--inter_data', default='F:/data-mm-bpr/Clothing/', help='Filename')

    parser.add_argument('--save_file', default='F:/mm-data/', help='Filename')
    parser.add_argument('--pre_dataset', default='Clothing/v5/pretrain/dim-100/', help='Filename')
    parser.add_argument('--fine_dataset', default='Clothing/v6/finetune/', help='Filename')
    parser.add_argument('--pre_train', default=False, help='train or not')
    parser.add_argument('--th', type=int, default=90, help='load which model.')
    parser.add_argument('--batch_valid', type=int, default=216, help='Batch size.')

    parser.add_argument('--user_num', type=int, default=39387, help='num of word.')
    parser.add_argument('--item_num', type=int, default=22986, help='num of word.')

    # textCNN 中参数
    # parser.add_argument('--word_num', type=int, default=10892, help='num of word.')
    parser.add_argument('--word_num', type=int, default=18032, help='num of word.')
    parser.add_argument('--word_dim', type=int, default=300, help='emb dim of word.')
    parser.add_argument('--img_dim', type=int, default=4096, help='num of word.')
    parser.add_argument('--pre_emb_dim', type=int, default=100, help='num of word.')
    parser.add_argument('--emb_dim', type=int, default=100, help='num of word.')

    parser.add_argument('--dropratio', type=float, default=0.5, help='text CNN drop ratio.')

    parser.add_argument('--pre_l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Learning rate.')
    parser.add_argument('--bpr_l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--finetune_l_r', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--neg_sam', type=int, default=10, help='Batch size.')

    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--pre_num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')

    '''
    # 数据集Yelp参数
    parser.add_argument('--data_path', default='F:/data-mm-bpr/Yelp/Yelp_item_list', help='Dataset path')
    parser.add_argument('--img_data_path',
                        default='F:/dataset/Yelp/20_item_vgg16_feature_dict',
                        help='Dataset path')
    parser.add_argument('--txt_data_path', default='F:/data-mm-bpr/Yelp/item_title_int_dict', help='Dataset path')
    parser.add_argument('--title_len', type=int, default=5, help='num of word.')
    parser.add_argument('--word_vec_arr', default='F:/data-mm-bpr/Yelp/word_int_vec_arr', help='Dataset path')
    parser.add_argument('--inter_data', default='F:/data-mm-bpr/Yelp/', help='Filename')

    parser.add_argument('--save_file', default='F:/mm-data/', help='Filename')
    parser.add_argument('--pre_dataset', default='Yelp/v5/pretrain/', help='Filename')
    parser.add_argument('--fine_dataset', default='Yelp/v6/finetune/', help='Filename')
    parser.add_argument('--pre_train', default=False, help='train or not')
    parser.add_argument('--th', type=int, default=90, help='load which model.')
    parser.add_argument('--batch_valid', type=int, default=216, help='Batch size.')

    parser.add_argument('--user_num', type=int, default=10457, help='num of word.')
    parser.add_argument('--item_num', type=int, default=8937, help='num of word.')

    # textCNN 中参数
    # parser.add_argument('--word_num', type=int, default=10892, help='num of word.')
    parser.add_argument('--word_num', type=int, default=5380, help='num of word.')
    parser.add_argument('--word_dim', type=int, default=300, help='emb dim of word.')
    parser.add_argument('--img_dim', type=int, default=4096, help='num of word.')
    parser.add_argument('--pre_emb_dim', type=int, default=50, help='num of word.')
    parser.add_argument('--emb_dim', type=int, default=50, help='num of word.')

    parser.add_argument('--dropratio', type=float, default=0.5, help='text CNN drop ratio.')

    parser.add_argument('--pre_l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Learning rate.')
    parser.add_argument('--bpr_l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--finetune_l_r', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--neg_sam', type=int, default=10, help='Batch size.')

    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--pre_num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_epoch', type=int, default=600, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    '''
    '''
    # food
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
    parser.add_argument('--bpr_l_r', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--finetune_l_r', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--neg_sam', type=int, default=10, help='Batch size.')

    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--pre_num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    '''
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
    train_dataset = TrainingDataset(new_item_list)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

    pre_train = PreTrain(args, intitem_fea_img, intitem_title_int, word_vec_arr)
    pre_train = pre_train.cuda()
    # 进行训练
    if args.pre_train is True:
        Train(pre_train, train_dataloader, args.pre_l_r, args.weight_decay, args.pre_num_epoch,
              args.save_file, args.pre_dataset, args.temperature)
    else:
        # 下游任务的dataloader
        user_item_dict = pk.load(open(args.inter_data + 'user_item_dict', 'rb'))
        train_list = pk.load(open(args.inter_data + 'train_interactions', 'rb'))
        test_dict = pk.load(open(args.inter_data + 'test_dict', 'rb'))
        valid_dict = pk.load(open(args.inter_data + 'valid_dict', 'rb'))

        bpr_train_dataset = bpr_TrainingDataset(train_list, user_item_dict, args.item_num, args.neg_sam)
        bpr_train_dataloader = DataLoader(bpr_train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        # 加载预训练模型
        model = Net(args, intitem_fea_img, intitem_title_int, word_vec_arr)
        model = model.cuda()

        # load_path = 'F:/mm-data/Clothing/v3/pretrain/450.pth'
        load_path = 'F:/mm-data/Clothing/v5/pretrain/dim-100/220.pth'
        # load_path = 'F:/mm-data/Yelp/v3/pretrain/260.pth'
        # load_path = 'F:/mm-data/food/v5/pretrain/140.pth'
        checkpoint = torch.load(load_path)
        model.PreTrain.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('加载 预训练模型 epoch {} 成功！'.format(start_epoch))

        # 240个epoch之后接着训练
        load_path = 'F:/mm-data/Clothing/v6/finetune/100.pth'
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('加载 模型 epoch {} 成功！'.format(start_epoch))

        BPR_Train(model, bpr_train_dataloader, args.bpr_l_r, args.finetune_l_r,
                  args.weight_decay, args.num_epoch, args.save_file, args.fine_dataset,
                  args.item_num, args.batch_valid, args.save_file, args.fine_dataset,
                  valid_dict, test_dict, user_item_dict, 1000)


if __name__ == '__main__':
    main()
