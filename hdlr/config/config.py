# -*- coding: utf-8 -*-
import numpy as np


class DefaultConfig:

    model = 'Multi_View'  # prior gru double attention network
    dataset = 'Gourmet_Food_data'

    norm_emb = False  # whether norm word embedding or not
    drop_out = 0.5

    # --------------optimizer---------------------#
    optimizer = 'Adam'
    weight_decay = 1e-4  # optimizer rameteri
    lr = 1e-3
    eps = 1e-8

    # -------------main.py-----------------------#
    seed = 2019
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []
    use_gpu = True   # user GPU or not
    num_epochs = 10  # the number of epochs for training
    num_workers = 20  # how many workers for loading data

    load_ckp = False
    ckp_path = ""
    fine_tune = True
    # ----------for confirmint the data -------------#
    use_word_embedding = True

    #  ----------id_embedding------------------------#
    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 100

    # --------------------CNN------------------------#
    r_filters_num = 100
    kernel_size = 3
    attention_size = 32
    att_method = 'matrix'
    review_weight = 'softmax'
    # -----------------gru/cnn----------------------#

    rm_merge = 'cat'
    ui_merge = 'cat'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    use_mask = False
    print_opt = 'def'
    prefer_user2v_path = './dataset/train/npy/'
    prefer_item2v_path = './dataset/train/npy/'
    pth_path = "checkpoints/Multi_View_Video_Ga_data_def.pth"  # the saved pth path for test
    

    use_word_drop = False

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        self.user_list = np.load(self.user_list_path, encoding='bytes')
        self.item_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_dict = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_dict = np.load(self.item2userid_path, encoding='bytes')
        self.ratingMatrix = np.load(self.ratingMatrix_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


class Office_Products_data_Config(DefaultConfig):
    data_root = './dataset/Office_Products_data/'
    w2v_path = './dataset/Office_Products_data/train/npy/w2v.npy'
    id2v_path = './dataset/Office_Products_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Office_Products_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Office_Products_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Office_Products_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Office_Products_data/train/npy/itemSummary2Index.npy'
    vocab_size = 42888
    word_dim = 300
    r_max_len = 248  # review max length
    s_max_len = 32  # summary max length

    train_data_size = 42611
    test_data_size = 10647
    user_num = 4905
    item_num = 2420
    user_mlp = [500, 80]
    item_mlp = [500, 80]
    batch_size = 100
    print_step = 200


class Gourmet_Food_data_Config(DefaultConfig):
    data_root = './dataset/Gourmet_Food_data/'
    w2v_path = './dataset/Gourmet_Food_data/train/npy/w2v.npy'

    user_list_path = './dataset/Gourmet_Food_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Gourmet_Food_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Gourmet_Food_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Gourmet_Food_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Gourmet_Food_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 74572
    word_dim = 300
    r_max_len = 168  # review max length

    u_max_r = 15
    i_max_r = 22

    train_data_size = 121003
    test_data_size = 15125
    user_num = 14681 + 2
    item_num = 8713 + 2
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 64
    print_step = 1000


class Video_Games_data_Config(DefaultConfig):
    data_root = './dataset/Video_Games_data/'
    w2v_path = './dataset/Video_Games_data/train/npy/w2v.npy'
    id2v_path = './dataset/Video_Games_data/matrix/npy/rawMatrix.npy'

    user_list_path = './dataset/Video_Games_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Video_Games_data/train/npy/itemReview2Index.npy'

    user_summary_path = './dataset/Video_Games_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Video_Games_data/train/npy/itemSummary2Index.npy'

    user2itemid_path = './dataset/Video_Games_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Video_Games_data/train/npy/item_user2id.npy'

    vocab_size = 194583
    word_dim = 300
    r_max_len = 517  # review max length
    s_max_len = 29   # summary max length

    train_data_size = 185439
    test_data_size = 23170
    user_num = 24303 + 2
    item_num = 10672 + 2
    u_max_r = 16
    i_max_r = 46
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 32
    print_step = 1000


class Toys_and_Games_data_Config(DefaultConfig):
    data_root = './dataset/Toys_and_Games_data/'
    w2v_path = './dataset/Toys_and_Games_data/train/npy/w2v.npy'

    user_list_path = './dataset/Toys_and_Games_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Toys_and_Games_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Toys_and_Games_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Toys_and_Games_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Toys_and_Games_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 79192
    word_dim = 300

    r_max_len = 183  # review max length

    train_data_size = 134104
    test_data_size = 33493
    user_num = 19412
    item_num = 11924
    u_max_r = 11
    i_max_r = 23
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 32
    print_step = 1000


class Kindle_Store_data_Config(DefaultConfig):
    data_root = './dataset/Kindle_Store_data/'
    w2v_path = './dataset/Kindle_Store_data/train/npy/w2v.npy'

    user_list_path = './dataset/Kindle_Store_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Kindle_Store_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Kindle_Store_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Kindle_Store_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Kindle_Store_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 278914
    word_dim = 300

    r_max_len = 211  # review max length

    train_data_size = 786159
    test_data_size = 98230
    user_num = 68223 + 2
    item_num = 61934 + 2
    u_max_r = 20
    i_max_r = 24
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 4
    print_step = 1000


class Movies_and_TV_data_Config(DefaultConfig):
    data_root = './dataset/Movies_and_TV_data/'
    w2v_path = './dataset/Movies_and_TV_data/train/npy/w2v.npy'

    user_list_path = './dataset/Movies_and_TV_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Movies_and_TV_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Movies_and_TV_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Movies_and_TV_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Movies_and_TV_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 764339
    word_dim = 300

    r_max_len = 326  # review max length

    train_data_size = 1358101
    test_data_size = 169716
    user_num = 123960
    item_num = 50052
    u_max_r = 16
    i_max_r = 49
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 16
    print_step = 5000


class Clothing_Shoes_and_Jewelry_data_Config(DefaultConfig):
    data_root = './dataset/Clothing_Shoes_and_Jewelry_data/'
    w2v_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/w2v.npy'
    id2v_path = './dataset/Clothing_Shoes_and_Jewelry_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/itemSummary2Index.npy'
    vocab_size = 67812
    word_dim = 300
    r_max_len = 97  # review max length
    s_max_len = 31  # summary max length

    train_data_size = 222984
    test_data_size = 55693
    user_num = 39387
    item_num = 23033
    user_mlp = [2000, 80]
    item_mlp = [4000, 80]
    batch_size = 80
    print_step = 1000


class Sports_and_Outdoors_data_Config(DefaultConfig):
    data_root = './dataset/Sports_and_Outdoors_data/'
    w2v_path = './dataset/Sports_and_Outdoors_data/train/npy/w2v.npy'
    id2v_path = './dataset/Sports_and_Outdoors_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Sports_and_Outdoors_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Sports_and_Outdoors_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Sports_and_Outdoors_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Sports_and_Outdoors_data/train/npy/itemSummary2Index.npy'
    vocab_size = 100129
    word_dim = 300
    r_max_len = 146  # review max length
    s_max_len = 29  # summary max length

    train_data_size = 237095
    test_data_size = 59242
    user_num = 35598
    item_num = 18357
    user_mlp = [2000, 80]
    item_mlp = [4000, 80]
    batch_size = 80
    print_step = 1000


class yelp2013_data_Config(DefaultConfig):
    data_root = './dataset/yelp2013_data'
    w2v_path = './dataset/yelp2013_data/train/npy/w2v.npy'

    user_list_path = './dataset/yelp2013_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/yelp2013_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/yelp2013_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/yelp2013_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/yelp2013_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 72807
    word_dim = 300

    r_max_len = 288

    u_max_r = 72
    i_max_r = 74

    train_data_size = 63172
    test_data_size = 7897

    user_num = 1631 + 2
    item_num = 1633 + 2

    batch_size = 16
    print_step = 1000


class yelp2014_data_Config(DefaultConfig):
    data_root = './dataset/yelp2014_data'
    w2v_path = './dataset/yelp2014_data/train/npy/w2v.npy'

    user_list_path = './dataset/yelp2014_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/yelp2014_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/yelp2014_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/yelp2014_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/yelp2014_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 126027
    word_dim = 300

    r_max_len = 284

    u_max_r = 72
    i_max_r = 85

    train_data_size = 184930
    test_data_size = 23116

    user_num = 4818 + 2
    item_num = 4194 + 2

    batch_size = 16
    print_step = 1000


class Video_Games_data_Config(DefaultConfig):
    data_root = './dataset/Video_Games_data'
    w2v_path = './dataset/Video_Games_data/train/npy/w2v.npy'

    user_list_path = './dataset/Video_Games_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Video_Games_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Video_Games_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Video_Games_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Video_Games_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 191668
    word_dim = 300

    r_max_len = 394

    u_max_r = 13
    i_max_r = 34

    train_data_size = 185439
    test_data_size = 23170

    user_num = 24303 + 2
    item_num = 10672 + 2

    batch_size = 32
    print_step = 1000

class Tucson_data_Config(DefaultConfig):
    data_root = './dataset/Tucson_data'
    w2v_path = './dataset/Tucson_data/train/npy/w2v.npy'

    user_list_path = './dataset/Tucson_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Tucson_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Tucson_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Tucson_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Tucson_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 84551
    word_dim = 300

    r_max_len = 198

    u_max_r = 33
    i_max_r = 32

    train_data_size = 150673
    test_data_size = 18572

    user_num = 8540 + 2
    item_num = 8867 + 2

    batch_size = 128
    print_step = 1000
    
    
class Tampa_data_Config(DefaultConfig):
    data_root = './dataset/Tampa_data'
    w2v_path = './dataset/Tampa_data/train/npy/w2v.npy'

    user_list_path = './dataset/Tampa_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Tampa_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Tampa_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Tampa_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Tampa_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 90121
    word_dim = 300

    r_max_len = 209

    u_max_r = 18
    i_max_r = 37

    train_data_size = 175937
    test_data_size = 21236

    user_num = 18437 + 2
    item_num = 8664 + 2

    batch_size = 128
    print_step = 1000
    
class Philadelphia_data_Config(DefaultConfig):
    data_root = './dataset/Philadelphia_data'
    w2v_path = './dataset/Philadelphia_data/train/npy/w2v.npy'

    user_list_path = './dataset/Philadelphia_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Philadelphia_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Philadelphia_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Philadelphia_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Philadelphia_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 90121
    word_dim = 300

    r_max_len = 209

    u_max_r = 18
    i_max_r = 37

    train_data_size = 175937
    test_data_size = 21236

    user_num = 18437 + 2
    item_num = 8664 + 2

    batch_size = 128
    print_step = 1000
    
    
class Video_Ga_data_Config(DefaultConfig):
    data_root = './pro_data/video_game'
    w2v_path = './pro_data/video_game/train/npy/w2v.npy'

    user_list_path = './pro_data/video_game/train/npy/userReview2Index.npy'
    item_list_path = './pro_data/video_game/train/npy/itemReview2Index.npy'

    user2itemid_path = './pro_data/video_game/train/npy/user_item2id.npy'
    item2userid_path = './pro_data/video_game/train/npy/item_user2id.npy'

    ratingMatrix_path = './pro_data/video_game/matrix/npy/ratingMatrix.npy'

    vocab_size = 128009
    word_dim = 300

    r_max_len = 336

    u_max_r = 27
    i_max_r = 20

    train_data_size = 170739
    test_data_size = 21078

    user_num = 10711 + 2
    item_num = 17005 + 2

    batch_size = 128
    print_step = 1000