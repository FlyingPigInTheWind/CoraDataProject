import itertools
import os
import pickle
import urllib.request
from collections import namedtuple

import numpy as np
import scipy.sparse as sp

Data = namedtuple('Data', ['x', 'y', 'adjacency',
'train_mask', 'val_mask', 'test_mask'])
#train_mask、val_mask、test_mask：与节点数相同的掩码，用于划分训练集、验证集、测试集

class CoraData(object):
    download_url="https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames=["{}".format(name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
    def __init__(self,data_root='cora',rebuild=False):
        self.data_root=data_root
        save_file=os.path.join(self.data_root,"processed_cora.pkl")
        if os.path.exists(save_file) and not rebuild:
            print("Using cached file:{}".format(save_file))
            self._data=pickle.load(save_file,"rb")#序列化
        else:
          #  self.maybe_download()
            self._data=self.process_data()
            with open(save_file,"wb") as f:
                pickle.dump(self.data,f)
            print("Cached file:{}".format(save_file))
    @property
    def data(self):
        return self._data
    def maybe_download(self):
        save_path=os.path.join(self.data_root,"raw")
        for name in self.filenames:
            if not os.path.exists(os.path.join(save_path,name)):
                test_url="{}/ind.cora.{}".format(self.download_url,name)
                print(test_url)
                self.download_data("{}/ind.cora.{}".format(self.download_url,name),save_path)
    @staticmethod
    def download_data(url,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data=urllib.request.urlopen(url)
        filename=os.path.basename(url)
        print("filename:{}".format(filename))
        with open(os.path.join(save_path,filename),'wb') as f:
            f.write(data.read())
        return True
    #数据处理
    def process_data(self):
        print("Processing data")
        x, tx, allx, y, ty, ally, graph, test_index=[
            self.read_data(
                os.path.join(self.data_root,"raw","ind.cora.{}".format(name)))for name in self.filenames]
        train_index=np.arange(y.shape[0])#shape（i）读取矩阵i+1维长度
        val_index=np.arange(y.shape[0],y.shape[0]+500)
        """
        np.arange()函数分为一个参数，两个参数，三个参数三种情况
            1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
            2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
            3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数

        """
        sorted_test_index=sorted(test_index)
        x=np.concatenate((allx,tx),axis=0)#数组拼接，参数为0纵向拼接，参数为1对应横向拼接
        y=np.concatenate((ally,ty),axis=0).argmax(axis=1)#横向按行依次取最大值所对应index
        print(test_index)
        print(sorted_test_index)
        x[test_index]=x[sorted_test_index]#?
        y[test_index]=y[sorted_test_index]
        num_nodes=x.shape[0]#节点数
        train_mask=np.zeros(num_nodes,dtype=np.bool)#返回不同形状不同类型0数组区分训练集、验证集、测试集
        val_mask=np.zeros(num_nodes,dtype=np.bool)
        test_mask=np.zeros(num_nodes,dtype=np.bool)
        train_mask[train_index]=True
        val_mask[val_index]=True
        test_mask[test_index]=True
        adjacency=self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())
        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        edge_index=[]
        num_nodes=len(adj_dict)
        for src,dst in adj_dict.items():
            edge_index.extend([src,v] for v in dst)
            edge_index.extend([v,src] for v in dst)
        # 由于上述得到的结果中存在重复的边，删掉这些重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))#通过k，若有重复边只取第一个
        edge_index = np.asarray(edge_index)#转换为Numpy数组
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")#稀疏矩阵coo_matrix((data, (row, col)), shape=(x, y))
        return adjacency

    @staticmethod
    def read_data(path):
        name=os.path.basename(path)#返回path最后文件名
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")#将文件中所有非空行按行以dtype类型创建数组表格数据
            return out
        else:
            out=pickle.load(open(path,"rb"),encoding="latin1")
            out=out.toarray() if hasattr(out,"toarray") else out#hasattr() 函数用于判断对象是否包含对应的属性。
            return out


