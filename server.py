#-*- coding:utf-8 -*-
import codecs
import yaml
import pickle
import tensorflow as tf
import numpy as np
import pdb
from load_data import load_vocs, init_data
from model import ClassficationModel
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from utils import map_item2id

class Model():

    def __init__(self):
        with open('./config.yml') as file_config:
            config = yaml.load(file_config)

        self.max_len = config['model_params']['sequence_length']
        feature_names = config['model_params']['feature_names']
        use_char_feature = config['model_params']['use_char_feature']

        # 初始化embedding shape, dropouts, 预训练的embedding也在这里初始化)
        feature_weight_shape_dict, feature_weight_dropout_dict, \
            feature_init_weight_dict = dict(), dict(), dict()
        for feature_name in feature_names:
            feature_weight_shape_dict[feature_name] = \
                config['model_params']['embed_params'][feature_name]['shape']
            feature_weight_dropout_dict[feature_name] = \
                config['model_params']['embed_params'][feature_name]['dropout_rate']
            path_pre_train = config['model_params']['embed_params'][feature_name]['path']
            if path_pre_train:
                with open(path_pre_train, 'rb') as file_r:
                    feature_init_weight_dict[feature_name] = pickle.load(file_r)
        # char embedding shape
        if use_char_feature:
            feature_weight_shape_dict['char'] = \
                config['model_params']['embed_params']['char']['shape']
            conv_filter_len_list = config['model_params']['conv_filter_len_list']
            conv_filter_size_list = config['model_params']['conv_filter_size_list']
        else:
            conv_filter_len_list = None
            conv_filter_size_list = None

        # 加载vocs
        path_vocs = []
        if use_char_feature:
            path_vocs.append(config['data_params']['voc_params']['char']['path'])
        for feature_name in feature_names:
            path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
        path_vocs.append(config['data_params']['voc_params']['label']['path'])
        self.vocs = load_vocs(path_vocs)

        # 加载模型
        self.model = ClassficationModel(
            sequence_length=config['model_params']['sequence_length'],
            nb_classes=config['model_params']['nb_classes'],
            nb_hidden=config['model_params']['bilstm_params']['num_units'],
            num_layers=config['model_params']['bilstm_params']['num_layers'],
            feature_weight_shape_dict=feature_weight_shape_dict,
            feature_init_weight_dict=feature_init_weight_dict,
            feature_weight_dropout_dict=feature_weight_dropout_dict,
            dropout_rate=config['model_params']['dropout_rate'],
            nb_epoch=config['model_params']['nb_epoch'], feature_names=feature_names,
            batch_size=config['model_params']['batch_size'],
            train_max_patience=config['model_params']['max_patience'],
            use_crf=config['model_params']['use_crf'],
            l2_rate=config['model_params']['l2_rate'],
            rnn_unit=config['model_params']['rnn_unit'],
            learning_rate=config['model_params']['learning_rate'],
            use_char_feature=use_char_feature,
            conv_filter_size_list=conv_filter_size_list,
            conv_filter_len_list=conv_filter_len_list,
            word_length=config['model_params']['word_length'],
            path_model=config['model_params']['path_model'])
        saver = tf.train.Saver()
        saver.restore(self.model.sess, config['model_params']['path_model'])

        self.label_voc = {}
        for key, value in self.vocs[-1].items():
            self.label_voc[value] = key

    def predict(self, query):
        items = [c for c in query]
        data_dict = {'f1':np.zeros((1, self.max_len))}
        data_dict['f1'][0,:] = map_item2id(items, self.vocs[0], self.max_len)
        res = self.model.predict(data_dict)
        return self.label_voc[res[0][0]+1]

model = Model()

class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        params = parse_qs(self.path[2:])
        string = params['query'][0]
        res = model.predict(string)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(res))
        self.end_headers()
        self.wfile.write(res.encode())

if __name__=='__main__':
    server_address = ('', 8080)
    server = HTTPServer(server_address, RequestHandler)
    server.serve_forever()
