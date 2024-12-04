

import torch
import json
import joblib
import pickle
import torch.utils.data as data_utils
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from model.ctr.inputs import SparseFeat, get_feature_names


def ctrdataset(path=None):
    
    if not path:
        return
    df = pd.read_csv(path, usecols=["user_id", "item_id", "click", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])
    df['video_category'] = df['video_category'].astype(str)
    df = sample_data(df)
    sparse_features = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]

    lbe = LabelEncoder()
    df['click'] = lbe.fit_transform(df['click'])

    print("Label Encoding ......")
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    # 输出a list of feature 属性：例如[SparseFeat(name='user_id', vocabulary_size=46572, embedding_dim=32, use_hash=False, dtype='int32', embedding_name='user_id', group_name='default_group'),
    # df[feat].nunique()：输出每个特征有多少个不同的值: vocabulary_size
    fixlen_feature_columns = [SparseFeat(feat, df[feat].nunique()) # 封装稀疏特征信息 的类。它主要用于推荐系统或深度学习任务中，用来定义特征的属性，并帮助构建模型的特征输入层。
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train, test = train_test_split(df, test_size=0.1)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    return train, test, train_model_input, test_model_input, linear_feature_columns, dnn_feature_columns

def sample_data(df):
    p_df = df[df.click.isin([1])]
    n_df = df[df.click.isin([0])]
    del df #节省内存
    n_df = n_df.sample(n=len(p_df)*2) #负样本采样，数量是正样本的2倍
   # df = p_df.append(n_df)
    df = pd.concat([p_df, n_df])
    del p_df, n_df #节省内存
    df = df.sample(frac=1) #如果frac=1，则表示抽取整个数据框的所有行，相当于对数据框进行重新洗牌。
    # 如果frac小于1，则表示抽取数据框中的一部分行，抽取的比例由frac参数指定。
    return df

