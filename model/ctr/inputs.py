
from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain

import torch
import torch.nn as nn
import numpy as np

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])): #namedtuple 提供了类字典形式的访问方式，但占用更少的内存。
    __slots__ = ()
    # name和vocabulary_size是必须的，其余的参数都有默认值
    def __new__(cls, name, vocabulary_size, embedding_dim=32, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME): #提供了默认参数和额外的验证逻辑
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = int(pow(vocabulary_size, 0.25))#6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()

def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    print(features)
    # OrderedDict({'user_id': (0, 1), 'item_id': (1, 2), 'video_category': (2, 3), 'gender': (3, 4), 'age': (4, 5), 'hist_1': (5, 6), 'hist_2': (6, 7), 'hist_3': (7, 8), 'hist_4': (8, 9), 'hist_5': (9, 10), 'hist_6': (10, 11), 'hist_7': (11, 12), 'hist_8': (12, 13), 'hist_9': (13, 14), 'hist_10': (14, 15)})
    return list(features.keys())


def build_input_features(feature_columns):
    """
    Return OrderedDict: {feature_name:(start, start+dimension)}
    """
    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        # SparseFeat：特征维度为1
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        # 
        # elif isinstance(feat, DenseFeat):
        #     features[feat_name] = (start, start + feat.dimension)
        #     start += feat.dimension
        # elif isinstance(feat, VarLenSparseFeat):
        #     features[feat_name] = (start, start + feat.maxlen)
        #     start += feat.maxlen
        #     if feat.length_name is not None and feat.length_name not in features:
        #         features[feat.length_name] = (start, start + 1)
        #         start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    # varlen_sparse_feature_columns = list(
    #     filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
        #  sparse_feature_columns + varlen_sparse_feature_columns}
        sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)

def combined_dnn_input(sparse_embedding_list):
    # if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
    #     sparse_dnn_input = torch.flatten(
    #         torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    #     dense_dnn_input = torch.flatten(
    #         torch.cat(dense_value_list, dim=-1), start_dim=1)
    #     return concat_fun([sparse_dnn_input, dense_dnn_input])
    if len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


