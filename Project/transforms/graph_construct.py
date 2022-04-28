# Copyright 2022 DSE lab.  All rights reserved.

import dgl
from dgl import nn as dglnn
from scipy.sparse import csc_matrix
import torch
import os
from collections import defaultdict
import pickle
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import numba
import scanpy as sc
from sklearn.preprocessing import normalize

import glob
import itertools
import random
from sklearn.neighbors import KDTree
import networkx as nx
from torch.nn import functional as F

import ..transforms.preprocess


# from dance.transforms.preprocess import ccaEmbed, l2norm, selectTopGenes


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def csr_cosine_similarity(input_csr_matrix):
    similarity = input_csr_matrix * input_csr_matrix.T
    square_mag = similarity.diagonal()
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    res = similarity.multiply(inv_mag).T.multiply(inv_mag)
    return res.toarray()


def cosine_similarity_gene(input_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    res = cosine_similarity(input_matrix)
    res = np.abs(res)
    return res


def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
    # beta to control the range of neighbourhood when calculate grey vale for one spot
    beta_half = round(beta / 2)
    g = []
    for i in range(len(x_pixel)):
        max_x = image.shape[0]
        max_y = image.shape[1]
        nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
              max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
        g.append(np.mean(np.mean(nbs, axis=0), axis=0))
    c0, c1, c2 = [], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
    return c3


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    # x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x) == len(x_pixel)) & (len(y) == len(y_pixel))
        print("Calculateing adj matrix using histology image...")
        # beta to control the range of neighbourhood when calculate grey vale for one spot
        # alpha to control the color scale
        beta_half = round(beta / 2)
        g = []
        for i in range(len(x_pixel)):
            max_x = image.shape[0]
            max_y = image.shape[1]
            nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
                  max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
            g.append(np.mean(np.mean(nbs, axis=0), axis=0))
        c0, c1, c2 = [], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0 = np.array(c0)
        c1 = np.array(c1)
        c2 = np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0), np.var(c1), np.var(c2))
        c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
        c4 = (c3 - np.mean(c3)) / np.std(c3)
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        z = c4 * z_scale
        z = z.tolist()
        print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xy only...")
        X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)


def construct_graph(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    return calculate_adj_matrix(x, y, x_pixel=x_pixel, y_pixel=y_pixel, image=image, beta=beta, alpha=alpha,
                                histology=histology)


def construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph=None, test=False, **kwargs):
    """Generate a feature-cell graph, enhanced with domain-knowledge (e.g. pathway).

    Parameters
    ----------
    u: torch.Tensor
        1-dimensional tensor. Cell node id of each cell-feature edge.
    v: torch.Tensor
        1-dimensional tensor. Feature node id of each cell-feature edge.
    e: torch.Tensor
        1-dimensional tensor. Weight of each cell-feature edge.
    cell_node_features: torch.Tensor
        1-dimensional or 2-dimensional tensor.  Node features for each cell node.
    enhance_graph: list[torch.Tensor]
        Node ids and edge weights of enhancement graph.

    Returns
    --------
    graph: DGLGraph
        The generated graph.
    """

    TRAIN_SIZE = kwargs['TRAIN_SIZE']
    FEATURE_SIZE = kwargs['FEATURE_SIZE']

    if enhance_graph is None:
        print('WARNING: Enhance graph disabled.')

    if kwargs['only_pathway'] and enhance_graph is not None:
        assert (kwargs['subtask'].find('rna') != -1)
        uu, vv, ee = enhance_graph

        graph_data = {
            ('feature', 'entail', 'cell'): (v, u),
            ('feature', 'pathway', 'feature'): (uu, vv),
        }
        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features

        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['entail'].data['weight'] = e
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

    elif kwargs['no_pathway'] or kwargs['subtask'].find('rna') == -1 or enhance_graph is None:
        if kwargs['inductive'] == 'opt':
            graph_data = {
                ('cell', 'occur', 'feature'): (u, v) if not test else (
                    u[:g.edges(etype='occur')[0].shape[0]], v[:g.edges(etype='occur')[0].shape[0]]),
                ('feature', 'entail', 'cell'): (v, u),
            }

        else:
            graph_data = {
                ('cell', 'occur', 'feature'): (u, v),
                ('feature', 'entail', 'cell'): (v, u),
            }

        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['entail'].data['weight'] = e
        graph.edges['occur'].data['weight'] = e[:graph.edges(etype='occur')[0].shape[0]]

    else:
        assert (kwargs['subtask'].find('rna') != -1)
        uu, vv, ee = enhance_graph

        if kwargs['inductive'] == 'opt':
            graph_data = {
                ('cell', 'occur', 'feature'): (u, v) if not test else (
                    u[:g.edges(etype='occur')[0].shape[0]], v[:g.edges(etype='occur')[0].shape[0]]),
                ('feature', 'entail', 'cell'): (v, u),
                ('feature', 'pathway', 'feature'): (uu, vv),
            }
        else:
            graph_data = {
                ('cell', 'occur', 'feature'): (u, v),
                ('feature', 'entail', 'cell'): (v, u),
                ('feature', 'pathway', 'feature'): (uu, vv),
            }
        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['entail'].data['weight'] = e
        graph.edges['occur'].data['weight'] = e[:graph.edges(etype='occur')[0].shape[0]]
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

    return graph


# TODO: haven't explained extra kwargs
def construct_pathway_graph(gex_data, **kwargs):
    """Generate nodes, edges and edge weights for pathway dataset.

    Parameters
    ----------
    gex_data: anndata.AnnData
        Gene data, contains feature matrix (.X) and feature names (.var['feature_types']).

    Returns
    --------
    uu: list[int]
        Predecessor node id of each edge.
    vv: list[int]
        Successor node id of each edge.
    ee: list[float]
        Edge weight of each edge.
    """

    pww = kwargs['pathway_weight']
    npw = kwargs['no_pathway']
    subtask = kwargs['subtask']
    pw_path = kwargs['pathway_path']
    uu = []
    vv = []
    ee = []

    assert (not npw)

    pk_path = f'pw_{subtask}_{pww}.pkl'
    #     pk_path = f'pw_{subtask}_{pww}.pkl'
    if os.path.exists(pk_path):
        print(
            'WARNING: Pathway file exist. Load pickle file by default. Auguments "--pathway_weight" and "--pathway_path" will not take effect.')
        uu, vv, ee = pickle.load(open(pk_path, 'rb'))
    else:
        # Load Original Pathway File
        with open(pw_path + '.entrez.gmt') as gmt:
            gene_list = gmt.read().split()

        gene_sets_entrez = defaultdict(list)
        indicator = 0
        for ele in gene_list:
            if not ele.isnumeric() and indicator == 1:
                indicator = 0
                continue
            if not ele.isnumeric() and indicator == 0:
                indicator = 1
                gene_set_name = ele
            else:
                gene_sets_entrez[gene_set_name].append(ele)

        with open(pw_path + '.symbols.gmt') as gmt:
            gene_list = gmt.read().split()

        gene_sets_symbols = defaultdict(list)

        for ele in gene_list:
            if ele in gene_sets_entrez:
                gene_set_name = ele
            elif not ele.startswith('http://'):
                gene_sets_symbols[gene_set_name].append(ele)

        pw = [i[1] for i in gene_sets_symbols.items()]

        # Generate New Pathway Data
        counter = 0
        total = 0
        feature_index = gex_data.var['feature_types'].index.tolist()
        gex_features = gex_data.X
        new_pw = []
        for i in pw:
            new_pw.append([])
            for j in i:
                if j in feature_index:
                    new_pw[-1].append(feature_index.index(j))

        if pww == 'cos':
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            sj = np.sqrt(
                                np.dot(gex_features[:, j].toarray().T, gex_features[:, j].toarray()).item())
                            sk = np.sqrt(
                                np.dot(gex_features[:, k].toarray().T, gex_features[:, k].toarray()).item())
                            jk = np.dot(gex_features[:, j].toarray().T, gex_features[:, k].toarray())
                            cossim = jk / sj / sk
                            ee.append(cossim.item())
        elif pww == 'one':
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            ee.append(1.)
        elif pww == 'pearson':
            corr = np.corrcoef(gex_features.toarray().T)
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            ee.append(corr[j][k])

        pickle.dump([uu, vv, ee], open(pk_path, 'wb'))

    # Apply Threshold
    pwth = kwargs['pathway_threshold']
    nu = []
    nv = []
    ne = []

    for i in range(len(uu)):
        if abs(ee[i]) > pwth:
            ne.append(ee[i])
            nu.append(uu[i])
            nv.append(vv[i])
    uu, vv, ee = nu, nv, ne

    return uu, vv, ee


def construct_basic_feature_graph(feature_mod1, feature_mod1_test=None, bf_input=None, device='cuda'):
    input_train_mod1 = csc_matrix(feature_mod1)

    if feature_mod1_test is not None:
        input_test_mod1 = csc_matrix(feature_mod1_test)
        assert (input_test_mod1.shape[1] == input_train_mod1.shape[1])

        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + input_train_mod1.shape[0]) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        sample_size = input_train_mod1.shape[0] + input_test_mod1.shape[0]
        weights = torch.from_numpy(
            np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()

    else:
        u = torch.from_numpy(
            np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
        sample_size = input_train_mod1.shape[0]
        weights = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data], axis=0)).float()

    graph_data = {
        ('cell', 'cell2feature', 'feature'): (u, v),
        ('feature', 'feature2cell', 'cell'): (v, u),
    }
    g = dgl.heterograph(graph_data)

    if bf_input:
        g.nodes['cell'].data['bf'] = gen_batch_features(bf_input)
    else:
        g.nodes['cell'].data['bf'] = torch.zeros(sample_size).float()

    g.nodes['cell'].data['id'] = torch.arange(sample_size).long()
    #     g.nodes['cell'].data['source'] =
    g.nodes['feature'].data['id'] = torch.arange(input_train_mod1.shape[1]).long()
    g.edges['cell2feature'].data['weight'] = g.edges['feature2cell'].data['weight'] = weights

    g = g.to(device)
    return g


def gen_batch_features(ad_inputs):
    """Generate statistical features for each batch in the input data, and assign batch features to each cell.
    This function returns batch features for each cell in all the input sub-datasets.

    Parameters
    ----------
    ad_inputs: list[anndata.AnnData]
        A list of AnnData object, each contains a sub-dataset.

    Returns
    --------
    batch_features: torch.Tensor
        A batch_features matrix, each row refers to one cell from the datasets. The matrix can be directly used as the
        node features of cell nodes.
    """

    cells = []
    columns = ['cell_mean', 'cell_std', 'nonzero_25%', 'nonzero_50%', 'nonzero_75%', 'nonzero_max', 'nonzero_count',
               'nonzero_mean', 'nonzero_std', 'batch']

    assert len(ad_inputs) < 10, "WARNING: Input of gen_bf_features should be a list of AnnData objects."

    for ad_input in ad_inputs:
        bcl = list(ad_input.obs['batch'])
        print(set(bcl))
        for i, cell in enumerate(ad_input.X):
            cell = cell.toarray()
            nz = cell[np.nonzero(cell)]
            cells.append([cell.mean(), cell.std(), np.percentile(nz, 25), np.percentile(nz, 50), np.percentile(nz, 75),
                          cell.max(), len(nz) / 1000, nz.mean(), nz.std(), bcl[i]])

    cell_features = pd.DataFrame(cells, columns=columns)
    batch_source = cell_features.groupby('batch').mean().reset_index()
    batch_list = batch_source.batch.tolist()
    batch_source = batch_source.drop('batch', axis=1).to_numpy().tolist()
    b2i = dict(zip(batch_list, range(len(batch_list))))
    batch_features = []

    for ad_input in ad_inputs:
        for b in ad_input.obs['batch']:
            batch_features.append(batch_source[b2i[b]])

    batch_features = torch.tensor(batch_features).float()

    return batch_features


def construct_modality_prediction_graph(dataset, **kwargs):
    """Construct the cell-feature graph object for modality prediction task, based on the input dataset.

    Parameters
    ----------
    dataset: datasets.multimodality.ModalityPredictionDataset
        The input dataset, typically includes four input AnnData sub-datasets, which are train_mod1, train_mod2,
        test_mod1 and test_mod2 respectively.

    Returns
    --------
    g: DGLGraph
        The generated graph.
    """

    train_mod1 = dataset.modalities[0]
    input_train_mod1 = dataset.sparse_features()[0]
    if kwargs['inductive'] == 'trans':
        input_test_mod1 = dataset.sparse_features()[2]

    CELL_SIZE = kwargs['CELL_SIZE']
    TRAIN_SIZE = kwargs['TRAIN_SIZE']

    if kwargs['cell_init'] == 'none':
        cell_node_features = torch.ones(CELL_SIZE).long()
    elif kwargs['cell_init'] == 'pca':
        embedder_mod1 = TruncatedSVD(n_components=100)
        X_train_np = embedder_mod1.fit_transform(input_train_mod1.toarray())
        X_test_np = embedder_mod1.transform(input_test_mod1.toarray())
        cell_node_features = torch.cat([torch.from_numpy(X_train_np), torch.from_numpy(X_test_np)], 0).float()
    if (not kwargs['no_pathway']) and (kwargs['subtask'].find('rna') != -1):
        enhance_graph = construct_pathway_graph(train_mod1, **kwargs)
    else:
        enhance_graph = None

    if kwargs['inductive'] != 'trans':
        u = torch.from_numpy(
            np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
        e = torch.from_numpy(input_train_mod1.tocsr().data).float()
        g = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, **kwargs)

        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + TRAIN_SIZE) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        e = torch.from_numpy(
            np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
        gtest = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, test=True, **kwargs)
        return g, gtest

    else:
        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + TRAIN_SIZE) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        e = torch.from_numpy(
            np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
        g = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, **kwargs)

        return g


def filter_data(X, highly_genes=500):
    """
    Remove less variable genes for single cell gene data

    Args:
        X ([type]): [description]
        highly_genes (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """

    X = np.ceil(X).astype(np.int)
    adata = sc.AnnData(X)

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4,
                                min_disp=0.5, n_top_genes=highly_genes, subset=True)
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)

    return genes_idx, cells_idx


def make_graph(X, Y=None, threshold=0, dense_dim=100, gene_data={},
               normalize_weights="log_per_cell", nb_edges=1,
               node_features="scale", same_edge_values=False,
               edge_norm=True):
    """
    Create DGL graph model from single cell gene data

    Args:
        X ([type]): [description]
        Y ([type], optional): [description]. Defaults to None.
        threshold (int, optional): [description]. Defaults to 0.
        dense_dim (int, optional): [description]. Defaults to 100.
        gene_data (dict, optional): [description]. Defaults to {}.
        normalize_weights (str, optional): [description]. Defaults to "log_per_cell".
        nb_edges (int, optional): [description]. Defaults to 1.
        node_features (str, optional): [description]. Defaults to "scale".
        same_edge_values (bool, optional): [description]. Defaults to False.
        edge_norm (bool, optional): [description]. Defaults to True.
    """
    num_genes = X.shape[1]

    graph = dgl.DGLGraph()
    gene_ids = torch.arange(X.shape[1], dtype=torch.int32).unsqueeze(-1)
    graph.add_nodes(num_genes, {'id': gene_ids})

    row_idx, gene_idx = np.nonzero(X > threshold)  # intra-dataset index

    if normalize_weights == "none":
        X1 = X
    if normalize_weights == "log_per_cell":
        X1 = np.log1p(X)
        X1 = X1 / (np.sum(X1, axis=1, keepdims=True) + 1e-6)

    if normalize_weights == "per_cell":
        X1 = X / (np.sum(X, axis=1, keepdims=True) + 1e-6)

    non_zeros = X1[(row_idx, gene_idx)]  # non-zero values

    cell_idx = row_idx + graph.number_of_nodes()  # cell_index
    cell_nodes = torch.tensor([-1] * len(X), dtype=torch.int32).unsqueeze(-1)

    graph.add_nodes(len(cell_nodes), {'id': cell_nodes})
    if nb_edges > 0:
        edge_ids = np.argsort(non_zeros)[::-1]
    else:
        edge_ids = np.argsort(non_zeros)
        nb_edges = abs(nb_edges)
        print(f"selecting weakest edges {int(len(edge_ids) * nb_edges)}")
    edge_ids = edge_ids[:int(len(edge_ids) * nb_edges)]
    cell_idx = cell_idx[edge_ids]
    gene_idx = gene_idx[edge_ids]
    non_zeros = non_zeros[edge_ids]

    if same_edge_values:
        graph.add_edges(
            gene_idx, cell_idx, {
                'weight':
                    torch.tensor(np.ones_like(non_zeros),
                                 dtype=torch.float32).unsqueeze(1)
            })
    else:
        graph.add_edges(
            gene_idx, cell_idx, {
                'weight':
                    torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1)
            })

    if node_features == "scale":
        nX = ((X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0))
        gene_feat = PCA(dense_dim, random_state=1).fit_transform(
            nX.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)
    if node_features == "scale_by_cell":
        nX = ((X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0))
        cell_feat = PCA(dense_dim, random_state=1).fit_transform(
            nX).astype(float)
        gene_feat = X1.T.dot(cell_feat).astype(float)
    if node_features == "none":
        gene_feat = PCA(dense_dim, random_state=1).fit_transform(
            X1.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)

    graph.ndata['features'] = torch.cat([torch.from_numpy(gene_feat),
                                         torch.from_numpy(cell_feat)],
                                        dim=0).type(torch.float)

    graph.ndata['order'] = torch.tensor([-1] * num_genes + list(np.arange(len(X))),
                                        dtype=torch.long)  # [gene_num+train_num]
    if Y is not None:
        graph.ndata['label'] = torch.tensor([-1] * num_genes + list(np.array(Y).astype(int)),
                                            dtype=torch.long)  # [gene_num+train_num]
    else:
        graph.ndata['label'] = torch.tensor(
            [-1] * num_genes + [np.nan] * len(X))
    nb_edges = graph.num_edges()

    if len(gene_data) != 0 and len(gene_data['gene1']) > 0:
        graph = external_data_connections(
            graph, gene_data, X, gene_idx, cell_idx)
    in_degrees = graph.in_degrees()
    # Edge normalization
    if edge_norm:
        for i in range(graph.number_of_nodes()):
            src, dst, in_edge_id = graph.in_edges(i, form='all')
            if src.shape[0] == 0:
                continue
            edge_w = graph.edata['weight'][in_edge_id]
            graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(
                edge_w)

    graph.add_edges(
        graph.nodes(), graph.nodes(), {
            'weight':
                torch.ones(graph.number_of_nodes(),
                           dtype=torch.float).unsqueeze(1)
        })
    return graph


############################
# For cell type deconvolution
############################

def kNN(data, k, query=None):
    tree = KDTree(data)
    if query is None:
        query = data
    dist, ind = tree.query(query, k)
    return dist, ind


# ' @param cell_embedding : pandas data frame
def KNN(cell_embedding, spots1, spots2, k):
    embedding_spots1 = cell_embedding.loc[spots1,]
    embedding_spots2 = cell_embedding.loc[spots2,]
    nnaa = kNN(embedding_spots1, k=k + 1)
    nnbb = kNN(embedding_spots2, k=k + 1)
    nnab = kNN(data=embedding_spots2, k=k, query=embedding_spots1)
    nnba = kNN(data=embedding_spots1, k=k, query=embedding_spots2)
    return nnaa, nnab, nnba, nnbb, spots1, spots2


def MNN(neighbors, colnames, num):
    max_nn = np.array([neighbors[1][1].shape[1], neighbors[2][1].shape[1]])
    if ((num > max_nn).any()):
        num = np.min(max_nn)
        # convert cell name to neighbor index
    spots1 = colnames
    spots2 = colnames
    nn_spots1 = neighbors[4]
    nn_spots2 = neighbors[5]
    cell1_index = [
        list(nn_spots1).index(i) for i in spots1 if (nn_spots1 == i).any()
    ]
    cell2_index = [
        list(nn_spots2).index(i) for i in spots2 if (nn_spots2 == i).any()
    ]
    ncell = range(neighbors[1][1].shape[0])
    ncell = np.array(ncell)[np.in1d(ncell, cell1_index)]
    # initialize a list
    mnn_cell1 = [None] * (len(ncell) * 5)
    mnn_cell2 = [None] * (len(ncell) * 5)
    idx = -1
    for cell in ncell:
        neighbors_ab = neighbors[1][1][cell, 0:5]
        mutual_neighbors = np.where(
            neighbors[2][1][neighbors_ab, 0:5] == cell)[0]
        for i in neighbors_ab[mutual_neighbors]:
            idx = idx + 1
            mnn_cell1[idx] = cell
            mnn_cell2[idx] = i
    mnn_cell1 = mnn_cell1[0:(idx + 1)]
    mnn_cell2 = mnn_cell2[0:(idx + 1)]
    import pandas as pd
    mnns = pd.DataFrame(np.column_stack((mnn_cell1, mnn_cell2)))
    mnns.columns = ['spot1', 'spot2']
    return mnns


def filterEdge(edges, neighbors, mats, features, k_filter):
    nn_spots1 = neighbors[4]
    nn_spots2 = neighbors[5]
    mat1 = mats.loc[features, nn_spots1].transpose()
    mat2 = mats.loc[features, nn_spots2].transpose()
    cn_data1 = dance.transforms.preprocess.l2norm(mat1)
    cn_data2 = dance.transforms.preprocess.l2norm(mat2)
    nn = kNN(data=cn_data2.loc[nn_spots2,],
             query=cn_data1.loc[nn_spots1,],
             k=k_filter)
    position = [
        np.where(
            edges.loc[:, "spot2"][x] == nn[1][edges.loc[:, 'spot1'][x],])[0]
        for x in range(edges.shape[0])
    ]
    nps = np.concatenate(position, axis=0)
    fedge = edges.iloc[nps,]
    return (fedge)


def stLinkGraph(  # count_list,
        # norm_list,
        scale_list,
        # features,
        combine,
        k_filter=200, num_cc=30):
    all_edges = []
    for row in combine:
        i = row[0]
        j = row[1]
        # counts1 = count_list[i]
        # counts2 = count_list[j]
        # norm_data1 = norm_list[i]
        # norm_data2 = norm_list[j]
        scale_data1 = scale_list[i]
        scale_data2 = scale_list[j]

        cell_embedding, loading = dance.transforms.preprocess.ccaEmbed(scale_data1,
                                                                       scale_data2,
                                                                       num_cc=num_cc)
        norm_embedding = dance.transforms.preprocess.l2norm(mat=cell_embedding[0])
        spots1 = scale_data1.columns
        spots2 = scale_data2.columns
        neighbor = KNN(cell_embedding=norm_embedding,
                       spots1=spots1,
                       spots2=spots2,
                       k=30)
        mnn_edges = MNN(neighbors=neighbor,
                        colnames=cell_embedding[0].index,
                        num=5)
        select_genes = dance.transforms.preprocess.selectTopGenes(Loadings=loading,
                                                                  dims=range(num_cc),
                                                                  DimGenes=100,
                                                                  maxGenes=200)
        Mat = pd.concat([scale_data1, scale_data2], axis=1)
        final_edges = filterEdge(edges=mnn_edges,
                                 neighbors=neighbor,
                                 mats=Mat,
                                 features=select_genes,
                                 k_filter=k_filter)
        final_edges['Dataset1'] = [i + 1] * final_edges.shape[0]
        final_edges['Dataset2'] = [j + 1] * final_edges.shape[0]
        all_edges.append(final_edges)
    return all_edges


def stLinkGraphConstruct(st_scale, st_label):
    N = len(st_scale)
    if (N == 1):
        combine = pd.Series([(0, 0)])
    else:
        combin = list(itertools.product(list(range(N)), list(range(N))))
        index = [i for i, x in enumerate([i[0] < i[1] for i in combin]) if x]
        combine = pd.Series(combin)[index]

    link1 = stLinkGraph(  # count_list=count_list,
        # norm_list=norm_list,
        scale_list=st_scale,
        # features=features,
        combine=combine)

    # ' ---- input data for link grpah 2 -----
    # files1 = glob.glob(path0 + "/ST_count/*.csv")
    # files1.sort()
    # tem_count = pd.read_csv(files1[1], index_col=0)
    tem_scale = st_scale[1]
    tem_scale.columns = tem_scale.columns.str.replace("mixt_", "rept_")

    tem_label = st_label[1]
    tem_label.columns = tem_label.columns.str.replace("mixt_", "rept_")

    # count_list2 = [count_list[1],tem_count]
    # norm_list2 = [norm_list[1], tem_norm]
    scale_list2 = [st_scale[1], tem_scale]

    link2 = stLinkGraph(  # count_list=count_list2,
        # norm_list=norm_list2,
        scale_list=scale_list2,
        # features=features,
        combine=combine,
        k_filter=100)

    graph1 = link1[0].iloc[:, 0:2].reset_index()
    graph1 = graph1.iloc[:, 1:3]
    # graph1.to_csv('./Datadir/Linked_graph1.csv')

    graph2 = link2[0].iloc[:, 0:2].reset_index()
    graph2 = graph2.iloc[:, 1:3]
    # graph2.to_csv('./Datadir/Linked_graph2.csv')

    # label1 = label_list[0]
    # label1.to_csv('./Datadir/Pseudo_Label1.csv', index=False)

    # label2 = label_list[1]
    # label2.to_csv('./Datadir/Real_Label2.csv', index=False)
    return ([graph1, graph2])


def stAdjConstruct(st_scale, st_label, data_train1, data_val1, data_test1, labels, lab_data2):
    # ' construct adjacent matrix
    graphs = stLinkGraphConstruct(st_scale, st_label)
    id_graph1 = graphs[0].copy()
    # id_graph1 = pd.read_csv('{}/Linked_graph1.csv'.format(data_dir),
    #                    index_col=0,
    #                    sep=',')

    # ' map index
    fake1 = np.array([-1] * len(lab_data2.index))
    index1 = np.concatenate((data_train1.index, fake1, data_val1.index,
                             data_test1.index)).flatten()
    # ' (feature_data.index==index1).all()
    fake2 = np.array([-1] * len(data_train1))
    fake3 = np.array([-1] * (len(data_val1) + len(data_test1)))
    find1 = np.concatenate((fake2, np.array(lab_data2.index), fake3)).flatten()

    row1 = [np.where(find1 == id_graph1.iloc[i, 1])[0][0]
            for i in range(len(id_graph1))
            ]
    col1 = [np.where(index1 == id_graph1.iloc[i, 0])[0][0]
            for i in range(len(id_graph1))
            ]
    adj = defaultdict(
        list)  # default value of int is 0
    for i in range(len(labels)):
        adj[i].append(i)
    for i in range(len(row1)):
        adj[row1[i]].append(col1[i])
        adj[col1[i]].append(row1[i])

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))
    return adj


############################
#          scDSC           #
############################
def construct_graph_sc(fname, features, label, method, topk):
    """Graph construction function for scDSC
    """
    num = len(label)
    if topk == None:
        topk = 0
    dist = None
    # Several methods of calculating the similarity relationship between samples i and j (similarity matrix Sij)
    if method == 'heat':
        dist = -0.5 * pair(features, metric='manhattan') ** 2
        dist = np.exp(dist)

    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)

    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    if topk:
        inds = []
        for i in range(dist.shape[0]):
            ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
            inds.append(ind)
    else:
        inds = dist

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()


def basic_feature_propagation(dataset, layers, transformed = True, device='cuda'):
    if transformed:
        g1 = construct_basic_feature_graph(dataset.transformed_features[0], dataset.transformed_features[2], device=device)
        g2 = construct_basic_feature_graph(dataset.transformed_features[1], dataset.transformed_features[3], device=device)
    else:
        g1 = construct_basic_feature_graph(dataset.sparse_features()[0], dataset.sparse_features()[2], device=device)
        g2 = construct_basic_feature_graph(dataset.sparse_features()[1], dataset.sparse_features()[3], device=device)

    hcell_mod1 = basic_feature_graph_propagation(g1, layers)
    hcell_mod2 = basic_feature_graph_propagation(g2, layers)

    return hcell_mod1, hcell_mod2


def basic_feature_graph_propagation(g, layers=3, alpha=0.5, beta=0.5, cell_init=None, feature_init='id', device='cuda', verbose=True):

    assert layers>2, 'Less than two feature graph propagation layers is equivalent to original features.'

    gconv = dglnn.HeteroGraphConv({
                    'cell2feature' : dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
                    'feature2cell' : dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
                },
            aggregate='sum')

    if feature_init is None:
        feature_X = torch.zeros((g.nodes('feature').shape[0], g.srcdata[cell_init]['cell'].shape[1])).to(device)
    elif feature_init == 'id':
        feature_X = F.one_hot(g.srcdata['id']['feature']).float().to(device)


    if cell_init is None:
        cell_X = torch.zeros(g.nodes('cell').shape[0], feature_X.shape[1]).to(device)
    else:
        cell_X = g.srcdata[cell_init]['cell']

    h = {'feature': feature_X,
         'cell': cell_X}
    hcell = []
    for i in range(layers):
        h1 = gconv(g, h, mod_kwargs={'cell2feature':{'edge_weight': g.edges['cell2feature'].data['weight']},
                                     'feature2cell':{'edge_weight': g.edges['feature2cell'].data['weight']}})
        if verbose: print(i, 'cell', h['cell'].abs().mean(), h1['cell'].abs().mean())
        if verbose: print(i, 'feature', h['feature'].abs().mean(), h1['feature'].abs().mean())

        h1['feature'] = (h1['feature'] - h1['feature'].mean())/(h1['feature'].std() if h1['feature'].mean()!=0 else 1)
        h1['cell'] = (h1['cell'] - h1['cell'].mean())/(h1['cell'].std() if h1['cell'].mean()!=0 else 1)

        h = {'feature': h['feature']*alpha + h1['feature']*(1-alpha),
             'cell': h['cell']*beta + h1['cell']*(1-beta)}

        h['feature'] = (h['feature'] - h['feature'].mean())/h['feature'].std()
        h['cell'] = (h['cell'] - h['cell'].mean())/h['cell'].std()

        hcell.append(h['cell'])

    if verbose: print(hcell[-1].abs().mean())

    return hcell[1:]
