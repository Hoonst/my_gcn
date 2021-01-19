import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

    return labels_onehot


def load_data(path="../data/cora", dataset="cora"):
    print(f"Loading {dataset} dataset...")

    idx_features_labels = np.genfromtxt(
        f"{path}/{dataset}.content", dtype=np.dtype(str)
    )
    #     array([['31336', '0', '0', ..., '0', '0', 'Neural_Networks'],
    #        ['1061127', '0', '0', ..., '0', '0', 'Rule_Learning'],
    #        ['1106406', '0', '0', ..., '0', '0', 'Reinforcement_Learning'],
    #        ...,
    #        ['1128978', '0', '0', ..., '0', '0', 'Genetic_Algorithms'],
    #        ['117328', '0', '0', ..., '0', '0', 'Case_Based'],
    #        ['24043', '0', '0', ..., '0', '0', 'Neural_Networks']],dtype='<U22')
    # idx_features_labels는 1:-1이 Features / -1이 Labels로 나타난다.
    # Graph를 구축하기 위한 관계도는 .cites에 위치

    # 각 Features를 csr_matrix(compressed Sparse Row Matrix)로 변환
    # 각 Labels를 One Hot Encoding으로 변환
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # csr_matrix:
    # coo_matrix: A sparse matrix in COOrdinate format

    # Build Graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # index for papers
    idx_map = {j: i for i, j in enumerate(idx)}                # index의 순차적 index

    edges_unordered = np.genfromtxt(
        f"{path}/{dataset}.cites", dtype=np.int32
    )                                                          # edges_unordered: 각 Paper가 가리키고 있는 Paper
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)                           # edges들을 고유 index가 아닌 개수에 맞는 index 변환
    
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # Build Symmetric Adjacency Matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # Normalize의 과정 속에 Degree의 -1/2승을 곱해주는 것이 있는데, 이를 위해
    # 먼저 Adjacency Matrix A에 eye matrix를 더해주는 Renormalization Trick을 적용하는 것을 볼 수 있습니다.
    
    idx_train = range(140)            # Train은 0~140
    idx_val = range(200, 500)         # Validation은 200~500
    idx_test = range(500, 1500)       # Test 는 500~1500 Index를 가진 데이터

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test