# Copyright 2022 DSE lab.  All rights reserved.
import anndata
import scipy.sparse as sp
from scipy.sparse import csr_matrix, vstack, load_npz, save_npz
from sklearn.decomposition import PCA
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.utils.extmath
import pandas as pd
import numpy as np
import scanpy as sc
import dgl
from dgl.sampling import random_walk, pack_traces
import dgl.function as fn
import torch
import torch.nn.functional as F
import collections
from pathlib import Path
from time import time
from ..transforms.graph_construct import stAdjConstruct
import time
import math
import os

import random
from sklearn.model_selection import train_test_split
import scipy


def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw=sc.pp.log1p(adata,copy=True) #check the rowname
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)

def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)

def normalize(adata,counts_per_cell_after=1e4,log_transformed=False):
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=counts_per_cell_after)

def log1p(adata):
    sc.pp.log1p(adata)

class tfidfTransformer():
    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        self.idf = X.shape[0] / X.sum(axis=0)
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            return tf.multiply(self.idf)
        else:
            tf = X / X.sum(axis=1, keepdims=True)
            return tf * self.idf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
 

class lsiTransformer():
    def __init__(self,
                 n_components: int = 20,
                 drop_first=True,
                ):
        
        self.drop_first=drop_first
        self.n_components = n_components+drop_first
        self.tfidfTransformer = tfidfTransformer()
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(n_components = self.n_components, random_state=777)
        self.fitted = None

    def fit(self, adata: anndata.AnnData):
        X = self.tfidfTransformer.fit_transform(adata.layers['counts'])
        X_norm = self.normalizer.fit_transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        self.pcaTransformer.fit(X_norm)
        self.fitted = True

    def transform(self, adata):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        X = self.tfidfTransformer.transform(adata.layers['counts'])
        X_norm = self.normalizer.transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = self.pcaTransformer.transform(X_norm)
        #         X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        #         X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
        lsi_df = pd.DataFrame(X_lsi, index=adata.obs_names).iloc[:, int(self.drop_first):]
        return lsi_df

    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)


####################################
# Copied from GraphSaint (https://github.com/lt610/GraphSaint)
####################################

# The base class of sampler
# (TODO): online sampling
class SAINTSampler(object):
    def __init__(self, dn, g, train_nid, node_budget, num_repeat=50):
        """
        :param dn: name of dataset
        :param g: full graph
        :param train_nid: ids of training nodes
        :param node_budget: expected number of sampled nodes
        :param num_repeat: number of times of repeating sampling one node
        """
        self.g = g
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_repeat = dn, num_repeat
        self.node_counter = torch.zeros((self.train_g.num_nodes(),))
        self.edge_counter = torch.zeros((self.train_g.num_edges(),))
        self.prob = None

        graph_fn, norm_fn = self.__generate_fn__()

        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            aggr_norm, loss_norm = np.load(norm_fn, allow_pickle=True)
        else:
            os.makedirs('./subgraphs/', exist_ok=True)

            self.subgraphs = []
            self.N, sampled_nodes = 0, 0

            t = time.perf_counter()
            while sampled_nodes <= self.train_g.num_nodes() * num_repeat:
                subgraph = self.__sample__()
                self.subgraphs.append(subgraph)
                sampled_nodes += subgraph.shape[0]
                self.N += 1
            print(f'Sampling time: [{time.perf_counter() - t:.2f}s]')
            np.save(graph_fn, self.subgraphs)

            t = time.perf_counter()
            self.__counter__()
            aggr_norm, loss_norm = self.__compute_norm__()
            print(f'Normalization time: [{time.perf_counter() - t:.2f}s]')
            np.save(norm_fn, (aggr_norm, loss_norm))

        self.train_g.ndata['l_n'] = torch.Tensor(loss_norm)
        self.train_g.edata['w'] = torch.Tensor(aggr_norm)
        self.__compute_degree_norm()

        self.num_batch = math.ceil(self.train_g.num_nodes() / node_budget)
        random.shuffle(self.subgraphs)
        self.__clear__()
        print("The number of subgraphs is: ", len(self.subgraphs))
        print("The size of subgraphs is about: ", len(self.subgraphs[-1]))

    def __clear__(self):
        self.prob = None
        self.node_counter = None
        self.edge_counter = None
        self.g = None

    def __counter__(self):

        for sampled_nodes in self.subgraphs:
            sampled_nodes = torch.from_numpy(sampled_nodes)
            self.node_counter[sampled_nodes] += 1

            subg = self.train_g.subgraph(sampled_nodes)
            sampled_edges = subg.edata[dgl.EID]
            self.edge_counter[sampled_edges] += 1

    def __generate_fn__(self):
        raise NotImplementedError

    def __compute_norm__(self):
        self.node_counter[self.node_counter == 0] = 1
        self.edge_counter[self.edge_counter == 0] = 1

        loss_norm = self.N / self.node_counter / self.train_g.num_nodes()

        self.train_g.ndata['n_c'] = self.node_counter
        self.train_g.edata['e_c'] = self.edge_counter
        self.train_g.apply_edges(fn.v_div_e('n_c', 'e_c', 'a_n'))
        aggr_norm = self.train_g.edata.pop('a_n')

        self.train_g.ndata.pop('n_c')
        self.train_g.edata.pop('e_c')

        return aggr_norm.numpy(), loss_norm.numpy()

    def __compute_degree_norm(self):

        self.train_g.ndata['train_D_norm'] = 1. / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        self.g.ndata['full_D_norm'] = 1. / self.g.in_degrees().float().clamp(min=1).unsqueeze(1)

    def __sample__(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batch:
            result = self.train_g.subgraph(self.subgraphs[self.n])
            self.n += 1
            return result
        else:
            random.shuffle(self.subgraphs)
            raise StopIteration()


class SAINTRandomWalkSampler(SAINTSampler):
    def __init__(self, num_roots, length, dn, g, train_nid, num_repeat=50):
        self.num_roots, self.length = num_roots, length
        super(SAINTRandomWalkSampler, self).__init__(dn, g, train_nid, num_roots * length, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}.npy'.format(self.dn, self.num_roots,
                                                                        self.length, self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots,
                                                                            self.length, self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        sampled_roots = torch.randint(0, self.train_g.num_nodes(), (self.num_roots, ))
        traces, types = random_walk(self.train_g, nodes=sampled_roots, length=self.length)
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()

############################
#Cell Type Annotation for ScDeepSort
#############################
def get_map_dict(map_path: Path, tissue):
    map_df = pd.read_excel(map_path / 'map.xlsx')
    # {num: {test_cell1: {train_cell1, train_cell2}, {test_cell2:....}}, num_2:{}...}
    map_dic = dict()
    for idx, row in enumerate(map_df.itertuples()):
        if getattr(row, 'Tissue') == tissue:
            num = getattr(row, 'num')
            test_celltype = getattr(row, 'Celltype')
            train_celltype = getattr(row, '_5')
            if map_dic.get(getattr(row, 'num')) is None:
                map_dic[num] = dict()
                map_dic[num][test_celltype] = set()
            elif map_dic[num].get(test_celltype) is None:
                map_dic[num][test_celltype] = set()
            map_dic[num][test_celltype].add(train_celltype)
    return map_dic


def normalize_weight(graph: dgl.DGLGraph):
    # normalize weight & add self-loop
    in_degrees = graph.in_degrees()
    for i in range(graph.number_of_nodes()):
        src, dst, in_edge_id = graph.in_edges(i, form='all')
        if src.shape[0] == 0:
            continue
        edge_w = graph.edata['weight'][in_edge_id]
        graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(edge_w)


def get_id_to_gene(gene_statistics_path):
    id2gene = []
    with open(gene_statistics_path, 'r', encoding='utf-8') as f:
        for line in f:
            id2gene.append(line.strip())
    return id2gene


def get_id_to_label(cell_statistics_path):
    id2label = []
    with open(cell_statistics_path, 'r', encoding='utf-8') as f:
        for line in f:
            id2label.append(line.strip())
    return id2label


def load_annotation_test_data(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    test = params.test_dataset
    tissue = params.tissue
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    species_data_path = proj_path / 'pretrained' / params.species
    statistics_path = species_data_path / 'statistics'

    if params.evaluate:
        map_path = proj_path / 'map' / params.species
        map_dict = get_map_dict(map_path, tissue)

    if not statistics_path.exists():
        statistics_path.mkdir()

    gene_statistics_path = statistics_path / (tissue + '_genes.txt')  # train+test gene
    cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')  # train labels

    # generate gene statistics file
    id2gene = get_id_to_gene(gene_statistics_path)
    # generate cell label statistics file
    id2label = get_id_to_label(cell_statistics_path)

    test_num = 0
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f"The build graph contains {num_genes} gene nodes with {num_labels} labels supported.")

    test_graph_dict = dict()  # test-graph dict
    if params.evaluate:
        test_label_dict = dict()  # test label dict
    test_index_dict = dict()  # test feature indices in all features
    test_mask_dict = dict()
    test_nid_dict = dict()
    test_cell_origin_id_dict = dict()

    ids = torch.arange(num_genes, dtype=torch.int32, device=device).unsqueeze(-1)

    # ==================================================
    # add all genes as nodes

    for num in test:
        test_graph_dict[num] = dgl.DGLGraph()
        test_graph_dict[num].add_nodes(num_genes, {'id': ids})
    # ====================================================

    matrices = []

    support_data = proj_path / 'pretrained' / f'{params.species}' / 'graphs' / f'{params.species}_{tissue}_data.npz'
    support_num = 0
    info = load_npz(support_data)
    print(f"load {support_data.name}")
    row_idx, gene_idx = np.nonzero(info > 0)
    non_zeros = info.data
    cell_num = info.shape[0]
    support_num += cell_num
    matrices.append(info)
    ids = torch.tensor([-1] * cell_num, device=device, dtype=torch.int32).unsqueeze(-1)
    total_cell = support_num

    for n in test:  # training cell also in test graph
        cell_idx = row_idx + test_graph_dict[n].number_of_nodes()
        test_graph_dict[n].add_nodes(cell_num, {'id': ids})
        test_graph_dict[n].add_edges(cell_idx, gene_idx,
                                     {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                             device=device).unsqueeze(1)})
        test_graph_dict[n].add_edges(gene_idx, cell_idx,
                                     {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                             device=device).unsqueeze(1)})

    for num in test:
        data_path = proj_path / params.test_dir / params.species / f'{params.species}_{tissue}{num}_data.{params.filetype}'
        if params.evaluate:
            type_path = proj_path / params.test_dir / params.species / f'{params.species}_{tissue}{num}_celltype.csv'
            # load celltype file then update labels accordingly
            cell2type = pd.read_csv(type_path, index_col=0)
            cell2type.columns = ['cell', 'type']
            cell2type['type'] = cell2type['type'].map(str.strip)
            # test_labels += cell2type['type'].tolist()
            test_label_dict[num] = cell2type['type'].tolist()

        # load data file then update graph
        if params.filetype == 'csv':
            df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        elif params.filetype == 'gz':
            df = pd.read_csv(data_path, compression='gzip', index_col=0)
        else:
            print(f'Not supported type for {data_path}. Please verify your data file')

        test_cell_origin_id_dict[num] = list(df.columns)
        df = df.transpose(copy=True)  # (cell, gene)

        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(f'{params.species}_{tissue}{num}_data.{params.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')
        tic = time()
        print(f'Begin to cumulate time of training/testing ...')
        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        # inter-dataset index
        cell_idx = row_idx + test_graph_dict[num].number_of_nodes()
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)

        # test_nodes_index_dict[num] = list(range(graph.number_of_nodes(), graph.number_of_nodes() + len(df)))
        ids = torch.tensor([-1] * len(df), device=device, dtype=torch.int32).unsqueeze(-1)
        test_index_dict[num] = list(range(num_genes + support_num + test_num, num_genes + support_num + test_num + len(df)))
        test_nid_dict[num] = list(
            range(test_graph_dict[num].number_of_nodes(), test_graph_dict[num].number_of_nodes() + len(df)))
        test_num += len(df)
        test_graph_dict[num].add_nodes(len(df), {'id': ids})
        # for the test cells, only gene-cell edges are in the test graph
        test_graph_dict[num].add_edges(gene_idx, cell_idx,
                                       {'weight': torch.tensor(non_zeros, dtype=torch.float32,
                                                               device=device).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(cell_idx)} edges.')
        total_cell += num

    support_index = list(range(num_genes + support_num))
    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:support_num].T)
    gene_feat = gene_pca.transform(sparse_feat[:support_num].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float).to(device)
    for num in test:
        test_graph_dict[num].ndata['features'] = features[support_index + test_index_dict[num]]

    for num in test:
        test_mask_dict[num] = torch.zeros(test_graph_dict[num].number_of_nodes(), dtype=torch.bool, device=device)
        test_mask_dict[num][test_nid_dict[num]] = 1
        test_nid_dict[num] = torch.tensor(test_nid_dict[num], dtype=torch.int64)
        # normalize weight & add self-loop
        normalize_weight(test_graph_dict[num])
        test_graph_dict[num].add_edges(test_graph_dict[num].nodes(), test_graph_dict[num].nodes(), {
            'weight': torch.ones(test_graph_dict[num].number_of_nodes(), dtype=torch.float, device=device).unsqueeze(
                1)})
        test_graph_dict[num].readonly()

    if params.evaluate:
        test_dict = {
            'graph': test_graph_dict,
            'label': test_label_dict,
            'nid': test_nid_dict,
            'mask': test_mask_dict,
            'origin_id': test_cell_origin_id_dict
        }
        time_used = time() - tic
        return total_cell, num_genes, num_labels, np.array(id2label, dtype=np.str), test_dict, map_dict, time_used
    else:
        test_dict = {
            'graph': test_graph_dict,
            'nid': test_nid_dict,
            'mask': test_mask_dict,
            'origin_id': test_cell_origin_id_dict
        }
        time_used = time() - tic
        return total_cell, num_genes, num_labels, np.array(id2label, dtype=np.str), test_dict, time_used
        

def get_id_2_gene(species_data_path, species, tissue, filetype):
    data_path = species_data_path
    data_files = data_path.glob(f'{species}_{tissue}*_data.{filetype}')
    genes = None
    for file in data_files:
        if filetype == 'csv':
            data = pd.read_csv(file, dtype=np.str, header=0).values[:, 0]
        else:
            data = pd.read_csv(file, compression='gzip', header=0).values[:, 0]
        if genes is None:
            genes = set(data)
        else:
            genes = genes | set(data)
    id2gene = list(genes)
    id2gene.sort()
    return id2gene


def get_id_2_label_and_label_statistics(species_data_path, species, tissue):
    data_path = species_data_path
    cell_files = data_path.glob(f'{species}_{tissue}*_celltype.csv')
    cell_types = set()
    cell_type_list = list()
    for file in cell_files:
        df = pd.read_csv(file, dtype=np.str, header=0)
        df['Cell_type'] = df['Cell_type'].map(str.strip)
        cell_types = set(df.values[:, 2]) | cell_types
        cell_type_list.extend(df.values[:, 2].tolist())
    id2label = list(cell_types)
    label_statistics = dict(collections.Counter(cell_type_list))
    return id2label, label_statistics


def save_statistics(statistics_path, id2label, id2gene, tissue):
    gene_path = statistics_path / f'{tissue}_genes.txt'
    label_path = statistics_path / f'{tissue}_cell_type.txt'
    with open(gene_path, 'w', encoding='utf-8') as f:
        for gene in id2gene:
            f.write(gene + '\r\n')
    with open(label_path, 'w', encoding='utf-8') as f:
        for label in id2label:
            f.write(label + '\r\n')


def load_annotation_data_internal(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    species = params.species
    tissue = params.tissue
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    proj_path = Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    species_data_path = proj_path / 'train' / species
    graph_path = proj_path / 'pretrained' / species / 'graphs'
    statistics_path = proj_path / 'pretrained' / species / 'statistics'
    
    if not species_data_path.exists():
        raise NotImplementedError

    if not statistics_path.exists():
        statistics_path.mkdir(parents=True)
    if not graph_path.exists():
        graph_path.mkdir(parents=True)

    # generate gene statistics file
    id2gene = get_id_2_gene(species_data_path, species, tissue, filetype=params.filetype)
    # generate cell label statistics file
    id2label, label_statistics = get_id_2_label_and_label_statistics(species_data_path, species, tissue)
    total_cell = sum(label_statistics.values())
    for label, num in label_statistics.items():
        if num / total_cell <= params.exclude_rate:
            id2label.remove(label)  # remove exclusive labels
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    save_statistics(statistics_path, id2label, id2gene, tissue)
    print(f"The build graph contains {num_genes} genes with {num_labels} labels supported.")

    graph = dgl.DGLGraph()

    gene_ids = torch.arange(num_genes, dtype=torch.int32, device=device).unsqueeze(-1)
    graph.add_nodes(num_genes, {'id': gene_ids})

    all_labels = []
    matrices = []
    num_cells = 0

    data_path = species_data_path
    data_files = data_path.glob(f'*{params.species}_{tissue}*_data.{params.filetype}')
    for data_file in data_files:
        number = ''.join(list(filter(str.isdigit, data_file.name)))
        type_file = species_data_path / f'{params.species}_{tissue}{number}_celltype.csv'

        # load celltype file then update labels accordingly
        cell2type = pd.read_csv(type_file, index_col=0)
        cell2type.columns = ['cell', 'type']
        cell2type['type'] = cell2type['type'].map(str.strip)
        cell2type['id'] = cell2type['type'].map(label2id)
        # filter out cells not in label-text
        filter_cell = np.where(pd.isnull(cell2type['id']) == False)[0]
        cell2type = cell2type.iloc[filter_cell]

        assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
        all_labels += cell2type['id'].tolist()

        # load data file then update graph
        # df = pd.read_csv(data_file, index_col=0)  # (gene, cell)

        if params.filetype == 'csv':
            df = pd.read_csv(data_file, index_col=0)  # (gene, cell)
        elif params.filetype == 'gz':
            df = pd.read_csv(data_file, compression='gzip', index_col=0)
        else:
            print(f'Not supported type for {data_path}. Please verify your data file')

        df = df.transpose(copy=True)  # (cell, gene)
        # filter out cells not in label-text
        df = df.iloc[filter_cell]
        assert cell2type['cell'].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]

        print(
            f'{params.species}_{tissue}{num}_data.{params.filetype} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')

        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        cell_idx = row_idx + graph.number_of_nodes()  # cell_index
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)

        num_cells += len(df)

        ids = torch.tensor([-1] * len(df), dtype=torch.int32, device=device).unsqueeze(-1)
        graph.add_nodes(len(df), {'id': ids})
        graph.add_edges(cell_idx, gene_idx,
                        {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})
        graph.add_edges(gene_idx, cell_idx,
                        {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})

        print(f'Added {len(df)} nodes and {len(cell_idx)} edges.')
        print(f'#Nodes in Graph: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')

    assert len(all_labels) == num_cells

    save_npz(graph_path / f'{params.species}_{tissue}_data', vstack(matrices))

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    assert sparse_feat.shape[0] == num_cells
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat.T)
    gene_feat = gene_pca.transform(sparse_feat.T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    print('------Train label statistics------')
    for i, label in enumerate(id2label, start=1):
        print(f"#{i} [{label}]: {label_statistics[label]}")

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    graph.ndata['features'] = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float).to(device)
    labels = torch.tensor([-1] * num_genes + all_labels, dtype=torch.long, device=device)  # [gene_num+train_num]

    # split train set and test set
    per = np.random.permutation(range(num_genes, num_genes + num_cells))
    test_ids = torch.tensor(per[:int(num_cells // ((1 - params.test_rate) / params.test_rate + 1))]).to(device)
    train_ids = torch.tensor(per[int(num_cells // ((1 - params.test_rate) / params.test_rate + 1)):]).to(device)

    # normalize weight
    normalize_weight(graph)
    # add self-loop
    graph.add_edges(graph.nodes(), graph.nodes(),
                    {'weight': torch.ones(graph.number_of_nodes(), dtype=torch.float, device=device).unsqueeze(1)})
    graph.readonly()

    return num_cells, num_genes, num_labels, graph, train_ids, test_ids, labels
########################################
#Cell Type Annotation for SVM
###########################################

def get_id_2_gene_svm(gene_statistics_path, species_data_path, tissue, train_dir: str):
    if not gene_statistics_path.exists():
        data_path = species_data_path / train_dir
        data_files = data_path.glob(f'*{tissue}*_data.csv')
        genes = None
        for file in data_files:
            data = pd.read_csv(file, dtype=np.str, header=0).values[:, 0]
            if genes is None:
                genes = set(data)
            else:
                genes = genes | set(data)
        id2gene = list(genes)
        id2gene.sort()
        with open(gene_statistics_path, 'w', encoding='utf-8') as f:
            for gene in id2gene:
                f.write(gene + '\r\n')
    else:
        id2gene = []
        with open(gene_statistics_path, 'r', encoding='utf-8') as f:
            for line in f:
                id2gene.append(line.strip())
    return id2gene


def get_id_2_label_svm(cell_statistics_path, species_data_path, tissue, train_dir: str):
    if not cell_statistics_path.exists():
        data_path = species_data_path / train_dir
        cell_files = data_path.glob(f'*{tissue}*_celltype.csv')
        cell_types = set()
        for file in cell_files:
            df = pd.read_csv(file, dtype=np.str, header=0)
            df['Cell_type'] = df['Cell_type'].map(str.strip)
            cell_types = set(df.values[:, 2]) | cell_types
            # cell_types = set(pd.read_csv(file, dtype=np.str, header=0).values[:, 2]) | cell_types
        id2label = list(cell_types)
        with open(cell_statistics_path, 'w', encoding='utf-8') as f:
            for cell_type in id2label:
                f.write(cell_type + '\r\n')
    else:
        id2label = []
        with open(cell_statistics_path, 'r', encoding='utf-8') as f:
            for line in f:
                id2label.append(line.strip())
    return id2label


def load_svm_data(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    train = params.train_dataset
    test = params.test_dataset
    tissue = params.tissue

    proj_path = Path(__file__).parent.resolve().parent.resolve()
    species_data_path = proj_path / 'data' / params.species
    statistics_path = species_data_path / 'statistics'
    map_dict = get_map_dict(species_data_path, tissue)

    gene_statistics_path = statistics_path / (tissue + '_genes.txt')  # train+test gene
    cell_statistics_path = statistics_path / (tissue + '_cell_type.txt')  # train labels

    # generate gene statistics file
    id2gene = get_id_2_gene_svm(gene_statistics_path, species_data_path, tissue, params.train_dir)
    # generate cell label statistics file
    id2label = get_id_2_label_svm(cell_statistics_path, species_data_path, tissue, params.train_dir)

    train_num, test_num = 0, 0
    # prepare unified genes
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    print(f"totally {num_genes} genes, {num_labels} labels.")

    train_labels = []
    test_label_dict = dict()  # test label dict
    test_index_dict = dict()  # test-num: [begin-index, end-index]
    test_cell_id_dict = dict()  # test-num: ['c1', 'c2'...]
    # TODO
    matrices = []

    for num in train + test:
        start = time()
        if num in train:
            data_path = species_data_path / (params.train_dir + f'/{params.species}_{tissue}{num}_data.csv')
            type_path = species_data_path / (params.train_dir + f'/{params.species}_{tissue}{num}_celltype.csv')
        else:
            data_path = species_data_path / (params.test_dir + f'/{params.species}_{tissue}{num}_data.csv')
            type_path = species_data_path / (params.test_dir + f'/{params.species}_{tissue}{num}_celltype.csv')

        # load celltype file then update labels accordingly
        cell2type = pd.read_csv(type_path, index_col=0)
        cell2type.columns = ['cell', 'type']
        cell2type['type'] = cell2type['type'].map(str.strip)
        if num in train:
            cell2type['id'] = cell2type['type'].map(label2id)
            assert not cell2type['id'].isnull().any(), 'something wrong in celltype file.'
            train_labels += cell2type['id'].tolist()
        else:
            # test_labels += cell2type['type'].tolist()
            test_label_dict[num] = cell2type['type'].tolist()

        # load data file then update graph
        df = pd.read_csv(data_path, index_col=0)  # (gene, cell)
        if num in test:
            test_cell_id_dict[num] = list(df.columns)
        df = df.transpose(copy=True)  # (cell, gene)

        assert cell2type['cell'].tolist() == df.index.tolist()
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]
        print(f'Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')
        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values

        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)

        if num in train:
            train_num += len(df)
        else:
            test_index_dict[num] = list(range(train_num + test_num, train_num + test_num + len(df)))
            test_num += len(df)
        print(f'Costs {time() - start:.3f} s in total.')
    train_labels = np.array(list(map(int, train_labels)))

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    test_feat_dict = dict()
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat[:train_num].T)
    gene_feat = gene_pca.transform(sparse_feat[:train_num].T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    # do normalization
    sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-6)
    # use weighted gene_feat as cell_feat
    cell_feat = sparse_feat.dot(gene_feat)  # [total_cell_num, d]
    train_cell_feat = cell_feat[:train_num]

    for num in test_label_dict.keys():
        test_feat_dict[num] = cell_feat[test_index_dict[num]]

    return num_labels, train_labels, train_cell_feat, map_dict, np.array(id2label, dtype=np.str), \
           test_label_dict, test_feat_dict, test_cell_id_dict

#######################################################
#For Single Cell ACTINN
#######################################################



# Get common genes, normalize  and scale the sets
def scale_sets(sets):
    # input -- a list of all the sets to be scaled
    # output -- scaled sets
    common_genes = set(sets[0].index)
    for i in range(1, len(sets)):
        common_genes = set.intersection(set(sets[i].index),common_genes)
    common_genes = sorted(list(common_genes))
    sep_point = [0]
    for i in range(len(sets)):
        sets[i] = sets[i].loc[common_genes,]
        sep_point.append(sets[i].shape[1])
    total_set = np.array(pd.concat(sets, axis=1, sort=False), dtype=np.float32)
    total_set = np.divide(total_set, np.sum(total_set, axis=0, keepdims=True)) * 10000
    total_set = np.log2(total_set+1)
    expr = np.sum(total_set, axis=1)
    total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
    cv = np.std(total_set, axis=1) / np.mean(total_set, axis=1)
    total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
    for i in range(len(sets)):
        sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]
    return sets



# Turn labels into matrix
def one_hot_matrix(labels, C):
    # input -- labels (true labels of the sets), C (# types)
    # output -- one hot matrix with shape (# types, # samples)
    C = torch.tensor(C)
    one_hot_matrix = torch.nn.functional.one_hot(labels, C)
    return one_hot_matrix

# Make types to labels dictionary
def type_to_label_dict(types):
    # input -- types
    # output -- type_to_label dictionary
    type_to_label_dict = {}
    all_type = list(set(types))
    for i in range(len(all_type)):
        type_to_label_dict[all_type[i]] = i
    return type_to_label_dict

# Convert types to labels
def convert_type_to_label(types, type_to_label_dict):
    # input -- list of types, and type_to_label dictionary
    # output -- list of labels
    types = list(types)
    labels = list()
    for type in types:
        labels.append(type_to_label_dict[type])
    return labels

# Function to create placeholders
def create_placeholders(n_x, n_y):
    X = torch.zeros(n_x)
    Y = torch.zeros(n_y)
    return X, Y

def load_actinn_data(train_set, train_label, test_set):
     
    train_set = pd.read_hdf(train_set, key="dge")
    train_set.index = [s.upper() for s in train_set.index]
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]
    train_label = pd.read_csv(train_label, header=None, sep="\t")
    test_set = pd.read_hdf(test_set, key="dge")
    test_set.index = [s.upper() for s in test_set.index]
    test_set = test_set.loc[~test_set.index.duplicated(keep='first')]
    barcode = list(test_set.columns)
    nt = len(set(train_label.iloc[:,1]))
    train_set, test_set = scale_sets([train_set, test_set])
    type_to_label_dict = type_to_label_dict(train_label.iloc[:,1])
    label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}
    print("Cell Types in training set:", type_to_label_dict)
    print("# Trainng cells:", train_label.shape[0])
    train_label = convert_type_to_label(train_label.iloc[:,1], type_to_label_dict)
    train_label = one_hot_matrix(train_label, nt)
    
    return train_set, train_label, test_set


#######################################################
#For Cell Type Deconvolution
#######################################################

def rowNormalizeFeatures(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for scGCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_pseudo_real_data(counts, log=True):
    for i in range(len(counts)):
        #norm_counts = sc.pp.normalize_total(counts[i])
        normalize(adata=counts[i],counts_per_cell_after=1e4)
        if log: 
            sc.pp.log1p(counts[i])
        

def get_pVal(counts, labels):
    sub_g = pd.DataFrame({'y': counts, 'x': labels})
    lm = ols('y ~ x', data=sub_g).fit()
    pval = sm.stats.anova_lm(lm, typ=1).loc[['x'], 'PR(>F)'][0]
    #print(pval)
    return(pval)

#' select variable genes
def select_var_genes_anova(counts,labels, nv=2000):
    
    # D: number of genes 
    D = counts.shape[1]
    
    #fit cell labels to cell expression, indidually for each gene 
    new_labels = labels['Cell Type']
    #for each gene (column), anova fit cell labels to the gene's expression across cells (rows)
    #get p-value corrected bonferroni for each gene 
    pv1 = [get_pVal(counts[:,d], new_labels) for d in range(D)]
        
    #get indices of nv genes with highest pVal 
    egen = sorted(range(len(pv1)), key=lambda i:pv1[i], reverse=True)[:nv]
    
    return(egen)


def gen_mix(sc_counts, nc_min=2, nc_max=10, umi_cutoff = 25000, downsample_counts=20000):
    mix_counts = sc_counts.copy()
    
    # sample between 2 and 10 cells randomly from the sc count matrix
    #sc.pp.subsample(mix_counts, n_obs=n_mix, random_seed=None)
    n_mix = random.choice(range(nc_min,nc_max+1))
    sample_inds = np.random.choice(10, size=n_mix, replace=False)
    mix_counts=mix_counts[sorted(sample_inds)]
    
    # Combine (sum) their transcriptomic info (counts)
    #downsample > 25k counts to <= 20k counts
    #if np.sum(mix_counts.X) > umi_cutoff:
    #    sc.pp.downsample_counts(mix_counts, total_counts=downsample_counts)
    mix_counts = np.sum(mix_counts.X, axis=0, keepdims=True)
    #obs = pd.DataFrame([[n_mix, np.sum(mix_counts), 0,0,0,0,0 ]], columns=pd.Index(['cell_count','total_umi_count', 'Bcell', 'Tcell', 
    #                                                                         'CAF', 'Macro', 'Endo']))
    obs = [n_mix, np.sum(mix_counts), 0,0,0,0,0 ]
    return(mix_counts, obs)

def gen_pseudo_spots(sc_counts, labels, clust_vr,nc_min=2, nc_max=10, N_p=1000, seed=0):
    np.random.seed(seed)
    tmp_sc_cnt = sc_counts.copy()

    mix_X = np.empty((0,tmp_sc_cnt.n_vars))
    mix_obs = []
    for i in range(N_p):
        #gets and combines a random mix of nc_min to nc_max cells
        mix_counts, obs = gen_mix(tmp_sc_cnt, umi_cutoff = 25000, downsample_counts=20000 )
        #append this mix to sample of pseudo mixtures
        mix_X = np.append(mix_X, mix_counts, axis=0)
        mix_obs += [obs]

    #create AnnData object with sample of pseudo mixtures (obs) 
    #annotations: cell counts, cell type compositions
    pseudo_counts = AnnData( X = mix_X,
                            obs = pd.DataFrame(mix_obs, index=pd.Index(['ps_mix_'+str(i+1) for i in range(N_p)]),
                                              columns=pd.Index(['cell_count','total_umi_count', 'Bcell', 'Tcell', 'CAF', 'Macro', 'Endo'])), 
                            var = sc_counts.var)
    return(pseudo_counts)
    


#' This function takes pseudo-spatail and real-spatial data to identify variable genes
def pseudo_spatial_process(counts, labels, scRNA=False, n_hvg=2000):
    #labels: sample (mix or spot) cell compositions 
    #counts: anndata type - one for scRNA or pseudo data, and one for real ST data
    st_counts = [ counts[0].copy(), counts[1].copy() ]
    #use common genes only
    genes1 = set(st_counts[0].var.index)
    genes2 = set(st_counts[1].var.index)
    intersect_genes=genes1.intersection(genes2)
    st_counts[0]=st_counts[0][:, list(intersect_genes)]
    st_counts[1]=st_counts[1][:, list(intersect_genes)]
    
    #if using scRNA data, generate pseudo ST data      
    if (scRNA):
        print('generate pseudo ST from scRNA')
        #get top nv variable genes in scRNA set
        sel_features = select_var_genes_anova(counts[0].X,labels[0], nv=n_hvg)
        #subset on top nv genes
        st_counts = [ st_counts[0][:, sel_features], st_counts[1][:, sel_features] ]
        
        #generate pseudo spots from scRNA (using variable genes) - as AnnData object
        st_counts = [ gen_pseudo_spots(st_counts[0], labels[0], N_p=500), st_counts[1] ]
        
    #otherwise, already using pseudo ST data -->  
    
    #library size normalization of the pseudo and real ST data
    #log indicates to apply log1p transformation after normalization
    normalize_pseudo_real_data(st_counts, log=True)
    
    #find highly variable genes for both pseudo and real data
    #flavors: seurat_v3 - expects raw count data
    #flavors: seurat (default), cell_ranger - expect log data
    batch_cnt = st_counts[0].concatenate(st_counts[1],index_unique=None)

    sc.pp.highly_variable_genes(batch_cnt, flavor='seurat',n_top_genes=n_hvg, batch_key='batch')


    hvgs=list(batch_cnt.var[batch_cnt.var.highly_variable==True].index)
    
    st_counts[0]._inplace_subset_var(np.array(batch_cnt.var.highly_variable==True))
    st_counts[1]._inplace_subset_var(np.array(batch_cnt.var.highly_variable==True))
    
    #scale/standardize pseudo and real ST data
    sc.pp.scale(st_counts[0])
    sc.pp.scale(st_counts[1])
    
    return([st_counts[0], st_counts[1]], labels, hvgs)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def split(st_data, lab_label, pre_process):
    data1 = pd.DataFrame(st_data[0].X,index=st_data[0].obs.index,
                columns=st_data[0].var.index)
    data2 = pd.DataFrame(st_data[1].X,index=st_data[1].obs.index,
                    columns=st_data[1].var.index)
    st_scale = [data1.transpose(), data2.transpose()]
    
    #data1 = data[0]
    #data2 = data[1]
    lab_label1 = lab_label[0]
    lab_label2 = lab_label[1]
    
    lab_data1 = data1.reset_index(drop=True)  #.transpose()
    lab_data2 = data2.reset_index(drop=True)  #.transpose()

    random.seed(123)
    p_data = lab_data1
    p_label = lab_label1

    temD_train, temd_test, temL_train, teml_test = train_test_split(
    p_data, p_label, test_size=0.1, random_state=1)
    temd_train, temd_val, teml_train, teml_val = train_test_split(
    temD_train, temL_train, test_size=0.1, random_state=1)

    print((temd_train.index == teml_train.index).all())
    print((temd_test.index == teml_test.index).all())
    print((temd_val.index == teml_val.index).all())
    data_train = temd_train
    label_train = teml_train
    data_test = temd_test
    label_test = teml_test
    data_val = temd_val
    label_val = teml_val

    data_train1 = data_train
    data_test1 = data_test
    data_val1 = data_val
    label_train1 = label_train
    label_test1 = label_test
    label_val1 = label_val

    train2 = pd.concat([data_train1, lab_data2])
    lab_train2 = pd.concat([label_train1, lab_label2])



    datas_train = np.array(train2)
    datas_test = np.array(data_test1)
    datas_val = np.array(data_val1)
    labels_train = np.array(lab_train2)
    labels_test = np.array(label_test1)
    labels_val = np.array(label_val1)

    #' convert pandas data frame to csr_matrix format
    datas_tr = scipy.sparse.csr_matrix(datas_train.astype(np.float64))
    datas_va = scipy.sparse.csr_matrix(datas_val.astype(np.float64))
    datas_te = scipy.sparse.csr_matrix(datas_test.astype(np.float64))

    M = len(data_train1)

    #' 4) get the feature object by combining training, test, valiation sets
    features = sp.vstack((sp.vstack((datas_tr, datas_va)), datas_te)).tolil()
    if pre_process: 
        features = rowNormalizeFeatures(features)
    
    labels_tr = labels_train
    labels_va = labels_val
    labels_te = labels_test

    labels = np.concatenate(
    [np.concatenate([labels_tr, labels_va]), labels_te])
    Labels = pd.DataFrame(labels)

    true_label = Labels

    #' new label with binary values
    new_label = labels
    idx_train = range(M)
    idx_pred = range(M, len(labels_tr))
    idx_val = range(len(labels_tr), len(labels_tr) + len(labels_va))
    idx_test = range(
    len(labels_tr) + len(labels_va),
    len(labels_tr) + len(labels_va) + len(labels_te))

    train_mask = sample_mask(idx_train, new_label.shape[0])
    pred_mask = sample_mask(idx_pred, new_label.shape[0])
    val_mask = sample_mask(idx_val, new_label.shape[0])
    test_mask = sample_mask(idx_test, new_label.shape[0])

    labels_binary_train = np.zeros(new_label.shape)
    labels_binary_val = np.zeros(new_label.shape)
    labels_binary_test = np.zeros(new_label.shape)
    labels_binary_train[train_mask, :] = new_label[train_mask, :]
    labels_binary_val[val_mask, :] = new_label[val_mask, :]
    labels_binary_test[test_mask, :] = new_label[test_mask, :]
    
    adj = stAdjConstruct(st_scale,lab_label, data_train1, data_val1, data_test1,labels, lab_data2)
    return adj, features, labels_binary_train, labels_binary_val, labels_binary_test, train_mask, pred_mask, val_mask, test_mask, new_label, true_label

def load_graph(path, data):
    '''Load graph for scDSC
    '''
    n = data.shape[0]
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = scipy.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # Construct a symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + scipy.eye(adj.shape[0])
    adj = scipy.sparse.coo_matrix(row_normalize(adj), dtype=np.float32)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj

def row_normalize(mx):
    # Row-normalize sparse matrix
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # Convert a scipy sparse matrix to a torch sparse tensor.
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def SVD(mat, num_cc):
    U, s, V = np.linalg.svd(mat)
    d = s[0:int(num_cc)]
    u = U[:, 0:int(num_cc)]
    v = V[0:int(num_cc), :].transpose()
    return u, v, d

def l2norm(mat):
    stat = np.sqrt(np.sum(mat**2, axis=1))
    cols = mat.columns
    mat[cols] = mat[cols].div(stat, axis=0)
    mat[np.isinf(mat)] = 0
    return mat

#' @param num.cc Number of canonical vectors to calculate
#' @param seed.use Random seed to set.
#' @importFrom SVD
def ccaEmbed(data1, data2,
          num_cc=30):
    random.seed(123)
    object1 = sklearn.preprocessing.scale(data1)
    object2 = sklearn.preprocessing.scale(data2)
    mat3 = np.matmul(np.matrix(object1).transpose(), np.matrix(object2))
    a = SVD(mat=mat3, num_cc=int(num_cc))
    embeds_data = np.concatenate((a[0], a[1]))
    ind = np.where(
        [embeds_data[:, col][0] < 0 for col in range(embeds_data.shape[1])])[0]
    embeds_data[:, ind] = embeds_data[:, ind] * (-1)

    embeds_data = pd.DataFrame(embeds_data)
    embeds_data.index = np.concatenate(
        (np.array(data1.columns), np.array(data2.columns)))
    embeds_data.columns = ['D_' + str(i) for i in range(num_cc)]
    d = a[2]

    cell_embeddings = np.matrix(embeds_data)
    combined_data = data1.merge(data2,
                                left_index=True,
                                right_index=True,
                                how='inner')
    new_data1 = combined_data.dropna()
    loadings = pd.DataFrame(np.matmul(np.matrix(new_data1), cell_embeddings))
    loadings.index = new_data1.index
    return [embeds_data, d], loadings

def checkFeature(data_use, features):
    data1 = data_use.loc[features, ]
    feature_var = data1.var(1)
    Var_features = features[np.where(feature_var != 0)[0]]
    return Var_features

def sortGenes(Loadings, dim, numG):
    data = Loadings.iloc[:, dim]
    num = np.round(numG / 2).astype('int')
    data1 = data.sort_values(ascending=False)
    data2 = data.sort_values(ascending=True)
    posG = np.array(data1.index[0:num])
    negG = np.array(data2.index[0:num])
    topG = np.concatenate((posG, negG))
    return topG

def SVD(mat, num_cc):
    U, s, V = np.linalg.svd(mat)
    d = s[0:int(num_cc)]
    u = U[:, 0:int(num_cc)]
    v = V[0:int(num_cc), :].transpose()
    return u, v, d

def l2norm(mat):
    stat = np.sqrt(np.sum(mat**2, axis=1))
    cols = mat.columns
    mat[cols] = mat[cols].div(stat, axis=0)
    mat[np.isinf(mat)] = 0
    return mat

#' @param num.cc Number of canonical vectors to calculate
#' @param seed.use Random seed to set.
#' @importFrom SVD
def ccaEmbed(data1, data2,
          num_cc=30):
    random.seed(123)
    object1 = sklearn.preprocessing.scale(data1)
    object2 = sklearn.preprocessing.scale(data2)
    mat3 = np.matmul(np.matrix(object1).transpose(), np.matrix(object2))
    a = SVD(mat=mat3, num_cc=int(num_cc))
    embeds_data = np.concatenate((a[0], a[1]))
    ind = np.where(
        [embeds_data[:, col][0] < 0 for col in range(embeds_data.shape[1])])[0]
    embeds_data[:, ind] = embeds_data[:, ind] * (-1)

    embeds_data = pd.DataFrame(embeds_data)
    embeds_data.index = np.concatenate(
        (np.array(data1.columns), np.array(data2.columns)))
    embeds_data.columns = ['D_' + str(i) for i in range(num_cc)]
    d = a[2]

    cell_embeddings = np.matrix(embeds_data)
    combined_data = data1.merge(data2,
                                left_index=True,
                                right_index=True,
                                how='inner')
    new_data1 = combined_data.dropna()
    loadings = pd.DataFrame(np.matmul(np.matrix(new_data1), cell_embeddings))
    loadings.index = new_data1.index
    return [embeds_data, d], loadings

def checkFeature(data_use, features):
    data1 = data_use.loc[features, ]
    feature_var = data1.var(1)
    Var_features = features[np.where(feature_var != 0)[0]]
    return Var_features

def sortGenes(Loadings, dim, numG):
    data = Loadings.iloc[:, dim]
    num = np.round(numG / 2).astype('int')
    data1 = data.sort_values(ascending=False)
    data2 = data.sort_values(ascending=True)
    posG = np.array(data1.index[0:num])
    negG = np.array(data2.index[0:num])
    topG = np.concatenate((posG, negG))
    return topG

def selectTopGenes(Loadings, dims, DimGenes, maxGenes):
    maxG = max(len(dims) * 2, maxGenes)
    gens = [None] * DimGenes
    idx = -1
    for i in range(1, DimGenes + 1):
        idx = idx + 1
        selg = []
        for j in dims:
            selg.extend(set(sortGenes(Loadings, dim=j, numG=i)))
        gens[idx] = set(selg)
    lens = np.array([len(i) for i in gens])
    lens = lens[lens < maxG]
    maxPer = np.where(lens == np.max(lens))[0][0] + 1
    selg = []
    for j in dims:
        selg.extend(set(sortGenes(Loadings, dim=j, numG=maxPer)))
    selgene = np.array(list(set(selg)), dtype=object)
    return (selgene)
