             # -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils
import types
from torch.autograd import Function
from losses import ContrastiveLoss
from attention import AttentionSimilarity
from cbam import CBAM
from CC_attention import CrissCrossAttention
# from AutomaticWeightedLoss import AutomaticWeightedLoss
from modelsHGNN import HGNN
import hypergraph_utils as hgut
from MFE import *

# np.random.seed(1337)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 32)
parser.add_argument("-c","--src_input_dim",type = int, default = 144)
parser.add_argument("-d","--tar_input_dim",type = int, default = 63) # PaviaU=103；salinas=204
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 15)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument("-m","--test_class_num",type=int, default=6)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=20, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)
def _init_():
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')
    if not os.path.exists('../classificationMap'):
        os.makedirs('../classificationMap')
_init_()

# load source domain data set
with open(os.path.join('../datasets',  'TRIAN_Houston_HSI_LIDAR_11_11.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
# print(source_imdb['Labels'])

# process source domain data set
data_train = np.array(source_imdb['data'] )# (77592, 9, 9, 128)
labels_train = np.array(source_imdb['Labels']) # 77592
print(data_train.shape)
print(labels_train.shape)

keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}  
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
# print(source_imdb['data'].shape) # (77592, 9, 9, 100)
source_imdb['data'] = np.array(source_imdb['data']).transpose((1, 2, 3, 0)) #(9, 9, 100, 77592)
source_imdb['Labels'] = np.array(source_imdb['Labels'])
source_imdb['set'] = np.array(source_imdb['set'])
print(source_imdb['data'].shape) # (77592, 9, 9, 100)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
test_data = '../datasets/Trento/Trento_HSI.mat'
test_label = '../datasets/Trento/Trento_gt.mat'
lid_data = '../datasets/Trento/Trento_LIDAR.mat'

Data_Band_Scaler, GroundTruth, Lidar_data = utils.load_onealldata(test_data, test_label, lid_data)
data_shape = GroundTruth.shape
print('**********************************************')
print(Data_Band_Scaler.shape) # (610, 340, 103)
print(GroundTruth.shape)
print(Lidar_data.shape)
Lidar_data = Lidar_data.T
[allRow, allColumn, allBand] = Data_Band_Scaler.shape
Lidar_data = Lidar_data.reshape(allRow, allColumn, 1)
Data_Band_Scaler = np.concatenate((Data_Band_Scaler, Lidar_data), axis=2)
# Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    print(GroundTruth.shape)
    # print( Lidar_data.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    # lidar_data =  utils.flip(Lidar_data)
    del Data_Band_Scaler
    del GroundTruth
    # del Lidar_data

    HalfWidth = 5
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    # lidar = lidar_data[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth
    # del lidar_data

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]#为什么不是x？
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):  #重复取5个样本，40次，5*40*9=1800
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    # imdb['Lidar'] = np.zeros([nTrain + nTest], dtype=np.float32)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)
        # imdb['lidar'][iSample] = lidar[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
        #                                  Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1]

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth,
                                                                     class_num=class_num, shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = next(iter(train_loader))
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain


# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)
        self.final_feat_dim = 160
        # self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM, bias=False)

    def forward(self, x): #x:(400,100,9,9)
        x = x.unsqueeze(1) # (400,1,100,9,9)
        x = self.block1(x) #(1,8,100,9,9)
        x = self.maxpool1(x) #(1,8,25,5,5)
        x = self.block2(x) #(1,16,25,5,5)
        x = self.maxpool2(x) #(1,16,7,3,3)
        x = self.conv(x) #(1,32,5,1,1)
        x = x.view(x.shape[0],-1) #(1,160)
        # y = self.classifier(x)
        return x


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x, eps=1e-5):
        # feature descriptor on the global spatial information
        # N, C, _, _ = x.size()
        # channel_center = x.view(N, C, -1)[:, :, int((x.shape[2] * x.shape[3] - 1) / 2)]
        # channel_center = channel_center.unsqueeze(2)
        # channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        # channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        # channel_std = channel_var.sqrt()
        # t = torch.cat((channel_mean, channel_std, channel_center), dim=2)
        #
        # y = self.conv(t.unsqueeze(1)).transpose(1, 2)
        # y = self.sigmoid(y)
        # x = x * (1 + y.expand_as(x))
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x



def conv3x3(in_channel, out_channel1, out_channel2):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel1,kernel_size=3, stride=3,padding=0,bias=False),
        nn.Conv3d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=3, stride=3, padding=0, bias=False),
        nn.BatchNorm3d(out_channel2),
        # nn.ReLU(inplace=True)
    )
    return layer


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.feature_encoder_y = conv3x3(1,8,16)
        self.final_feat_dim = FEATURE_DIM  # 64+32
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.cbam_y = CBAM(21)
        self.mfe = MFE(1, 1, 6, 11)
        self.cc_x = CrissCrossAttention(100)
        self.cc_y = CrissCrossAttention(21)
    def forward(self, x, y, domain='source'):  # x
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)


        x = x.unsqueeze(1)
        feature_fuse = self.mfe(x, y)


        return feature_fuse, feature_fuse



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())



crossEntropy = nn.CrossEntropyLoss().cuda()
attention_module = AttentionSimilarity(hidden_size=160) # hidden_size depends on the encoder
contrast_criterion = ContrastiveLoss(temperature=10) # inverse temp is used (0.1)
# awl = AutomaticWeightedLoss(2)

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


# run 10 times
nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
# seeds = [1336]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,GT,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = Network()
    feature_encoder.apply(weights_init)
    feature_encoder.cuda()
    feature_encoder.train()


    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)

    model_ft = HGNN(in_ch=FEATURE_DIM,
                        n_class=FEATURE_DIM,
                        n_hid=64,
                        dropout=0)
    model_ft = model_ft.cuda()

    model_ft.train()
    model_ft_optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001,
                                              weight_decay=0.0005)
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    schedular = torch.optim.lr_scheduler.MultiStepLR(model_ft_optimizer,
                                                     milestones=[100],
                                                     gamma=0.9)
    # schedular = torch.optim.lr_scheduler.StepLR(model_ft_optimizer,
    #                                             step_size=100,
    #                                                  gamma=0.5)
    schedular.step()
    # feature_encoder_optim = torch.optim.Adam(parameters, lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []
    test =[]
    label = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(20000):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = next(source_iter)
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = next(source_iter)

        try:
            target_data, target_label = next(target_iter)
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = next(target_iter)

        # source domain few-shot + domain adaptation
        if episode < 100:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = next(iter(support_dataloader))  # (5, 100, 9, 9)
            querys, query_labels = next(iter(query_dataloader))  # (75,100,9,9)
            # calculate features
            supports_HS = supports[:, :144, :, :]
            supports_lidar = supports[:, 144:, :, :]
            querys_HS = querys[:, :144, :, :]
            querys_lidar = querys[:, 144:, :, :]
            # supports_lidar = supports_lidar.unsqueeze(1)
            # querys_lidar = querys_lidar.unsqueeze(1)

            # calculate features
            support_features, support_outputs = feature_encoder(supports_HS.cuda(), supports_lidar.cuda(),
                                                                domain='source')  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_outputs = feature_encoder(querys_HS.cuda(), querys_lidar.cuda(),
                                                            domain='source')  # torch.Size([409, 32, 7, 3, 3])

            all_features = torch.cat((support_features, query_features), 0)
            all_labels = torch.cat((support_labels, query_labels), 0)

            all_features = torch.Tensor(all_features).to(device)
            all_labels = torch.Tensor(all_labels).squeeze().long().to(device)

            H = None
            tmp = hgut.construct_H_with_KNN(all_features, K_neigs=5,
                                            split_diff_scale=False,
                                            is_probH=True, m_prob=1)
            H = hgut.hyperedge_concat(H, tmp)

            G = hgut.generate_G_from_H(H)

            G = torch.Tensor(G).to(device)

            HGNN_features = model_ft(all_features, G)

            support_features_1 = HGNN_features[: support_features.shape[0], :]
            query_features_1 = HGNN_features[support_features.shape[0]:, :]

            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features_1.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features_1

            '''fsl_loss'''

            logits = euclidean_metric(query_features_1, support_proto)
            f_loss = crossEntropy(logits, query_labels.long().cuda())

            '''contrastive loss'''

            all_features_1 = torch.cat((support_features_1, query_features_1), 0)
            all_labels = torch.cat((support_labels, query_labels), 0)
            all_labels = all_labels.cuda()
            con_loss = contrast_criterion(all_features_1, labels=all_labels)

            # total_loss = fsl_loss + domain_loss
            # loss = f_loss# 0.01
            loss =0.1*con_loss + 10*f_loss
            # loss = awl(loss_contrast, f_loss)

            # Update parameters
            feature_encoder.zero_grad()
            model_ft_optimizer.zero_grad()
            # domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            model_ft_optimizer.step()
            # domain_classifier_optim.step()

            # total_hit += torch.sum(torch.argmax(outputs, dim=1).cpu() == all_labels.cpu()).item()
            # total_num += all_features.shape[0]
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # print('*****************************************************')
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
            # sample datas
            supports, support_labels = next(iter(support_dataloader))  # (5, 100, 9, 9)
            querys, query_labels = next(iter(query_dataloader))  # (75,100,9,9)
            # print(supports.shape)
            supports_HS = supports[:, :63, :, :]
            supports_lidar = supports[:,63:,:,:]
            querys_HS = querys[:,:63,:,:]
            querys_lidar = querys[:, 63:,:,:]
            # supports_lidar = supports_lidar.unsqueeze(1)
            # querys_lidar = querys_lidar.unsqueeze(1)
            # calculate features
            support_features, support_outputs = feature_encoder(supports_HS.cuda(), supports_lidar.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_outputs = feature_encoder(querys_HS.cuda(), querys_lidar.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])

            all_features = torch.cat((support_features, query_features), 0)
            all_labels = torch.cat((support_labels, query_labels), 0)

            all_features = torch.Tensor(all_features).to(device)
            all_labels = torch.Tensor(all_labels).squeeze().long().to(device)

            H = None
            tmp = hgut.construct_H_with_KNN(all_features, K_neigs=5,
                                            split_diff_scale=False,
                                            is_probH=True, m_prob=1)
            H = hgut.hyperedge_concat(H, tmp)

            G = hgut.generate_G_from_H(H)

            G = torch.Tensor(G).to(device)

            HGNN_features = model_ft(all_features, G)

            support_features_1 = HGNN_features[: support_features.shape[0], :]
            query_features_1 = HGNN_features[support_features.shape[0]:, :]

            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features_1.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features_1

            '''fsl_loss'''

            logits = euclidean_metric(query_features_1, support_proto)
            f_loss = crossEntropy(logits, query_labels.long().cuda())

            '''contrastive loss'''

            all_features_1 = torch.cat((support_features_1, query_features_1), 0)
            all_labels = torch.cat((support_labels, query_labels), 0)
            all_labels = all_labels.cuda()
            con_loss = contrast_criterion(all_features_1, labels=all_labels)

            # total_loss = fsl_loss + domain_loss
            # loss = f_loss# 0.01
            loss = 1*con_loss + 0.1*f_loss
            # loss = awl(loss_contrast, f_loss)

            # Update parameters
            feature_encoder.zero_grad()
            model_ft_optimizer.zero_grad()
            # domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            model_ft_optimizer.step()
            # domain_classifier_optim.step()

            # total_hit += torch.sum(torch.argmax(outputs, dim=1).cpu() == all_labels.cpu()).item()
            # total_num += all_features.shape[0]
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}:  fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                          f_loss.item(),
                                                                                          total_hit / total_num,
                                                                                          loss.item()))

        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            model_ft.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)


            train_datas, train_labels = next(iter(train_loader))
            train_datas_HS = train_datas[:, :63, :, :]
            train_datas_lidar = train_datas[:, 63:, :, :]
            # train_datas_lidar = train_datas_lidar.unsqueeze(1)


            train_features, _ = feature_encoder(Variable(train_datas_HS).cuda(), Variable(train_datas_lidar).cuda(), domain='target')  # (45, 160)

            H = None
            tmp = hgut.construct_H_with_KNN(train_features, K_neigs=5,
                                            split_diff_scale=False,
                                            is_probH=True, m_prob=1)
            H = hgut.hyperedge_concat(H, tmp)

            G = hgut.generate_G_from_H(H)
            G = torch.Tensor(G).to(device)

            train_features = model_ft(train_features, G)


            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            # print(max_value.item())
            # print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()

            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_datas_HS = test_datas[:, :63, :, :]
                test_datas_lidar = test_datas[:, 63:, :, :]

                # test_datas_lidar = test_datas_lidar.unsqueeze(1)

                test_features, _ = feature_encoder(Variable(test_datas_HS).cuda(), Variable(test_datas_lidar).cuda(), domain='target')  # (100, 160)

                #test_features = (test_features - min_value) * 1.0 / (max_value - min_value)

                # test.append(test_features)
                # label.append(test_labels)
                H = None
                tmp = hgut.construct_H_with_KNN(test_features, K_neigs=5,
                                                split_diff_scale=False,
                                                is_probH=True, m_prob=1)
                H = hgut.hyperedge_concat(H, tmp)

                G = hgut.generate_G_from_H(H)
                G = torch.Tensor(G).to(device)

                test_features = model_ft(test_features, G)


                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str( "checkpoints/DFSL_feature_encoder_" + "UP_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))


    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = GT, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))


print('***********************************************************************************')
# print('每一次每一类的结果：')
# print(100 * A)
# print('***********************************************************************************')
# print('每一次每一类的Kappa：')
# print(100*k)
# print('***********************************************************************************')
AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(TEST_CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

# #################classification map################################
#
for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]
        # if best_G[i][j] == 16:
        #     hsi_pic[i, j, :] = [0.5, 0.75, 1]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/H-Trento_MMHGNN_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
