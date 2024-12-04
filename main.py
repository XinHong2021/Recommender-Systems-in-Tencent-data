from utils import *
from model.ctr.wdl import WDL

def get_data(args):
    data_path = args["data_path"]
    train, test, train_model_input, test_model_input, lf_columns, df_columns = ctrdataset(data_path)
    return train, test, train_model_input, test_model_input, lf_columns, df_columns


def get_model(args, linear_feature_columns=None, dnn_feature_columns=None, history_feature_list=None):
    name = args["model"]
    if name == 'wdl':
        return WDL(linear_feature_columns, dnn_feature_columns, task='binary', device=args["device"])
    else:
        raise ValueError('unknown model name: ' + name)