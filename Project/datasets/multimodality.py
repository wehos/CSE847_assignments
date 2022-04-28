from ..data import *
import anndata as ad
import pickle
from dance.transforms.preprocess import lsiTransformer

class MultiModalityDataset():
    def __init__(self, task, data_url, subtask, data_dir = "./data"):

        assert (subtask in ['openproblems_bmmc_multiome_phase2_mod2', 'openproblems_bmmc_multiome_phase2_rna',
                            'openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_cite_phase2_mod2',
                            'openproblems_bmmc_cite_phase2', 'openproblems_bmmc_multiome_phase2',
                            'adt2gex', 'gex2adt',
                            'atac2gex', 'gex2atac']), 'Undefined subtask.'

        assert (task in ['predict_modality', 'match_modality', 'joint_embedding']), 'Undefined task.'

        # regularize subtask name
        if task == 'joint_embedding':
            if subtask.find('adt')!=-1: subtask = 'openproblems_bmmc_cite_phase2'
            else: subtask = 'openproblems_bmmc_multiome_phase2'
        else:
            if subtask == 'adt2gex':
                subtask = 'openproblems_bmmc_cite_phase2_mod2'
            elif subtask == 'gex2adt':
                subtask = 'openproblems_bmmc_cite_phase2_rna'
            elif subtask == 'atac2gex':
                subtask = 'openproblems_bmmc_multiome_phase2_mod2'
            elif subtask == 'gex2atac':
                subtask = 'openproblems_bmmc_multiome_phase2_rna'

        self.task = task
        self.subtask = subtask
        self.data_dir = data_dir
        self.loaded = False
        self.data_url = data_url

    def download_data(self):
        # download data
        download_file(self.data_url, self.data_dir+"/{}.zip".format(self.subtask))
        unzip_file(self.data_dir+"/{}.zip".format(self.subtask), self.data_dir)
        return self

    def is_complete(self):
        # judge data is complete or not
        if self.task == 'joint_embedding':
            return os.path.exists(
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_mod1.h5ad')
            ) and os.path.exists(
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_mod2.h5ad'))
        else:
            return os.path.exists(
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_train_mod1.h5ad')
            ) and os.path.exists(
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_train_mod2.h5ad')
            ) and os.path.exists(
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_test_mod1.h5ad')
            ) and os.path.exists(
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_test_mod2.h5ad'))


    def load_data(self):
        # Load data from existing h5ad files, or download files and load data.
        if self.is_complete():
            pass
        else:
            self.download_data()
            assert self.is_complete()

        if self.task == 'joint_embedding':
            mod_path_list = [os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_mod1.h5ad'),
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_mod2.h5ad')]
        else:
            mod_path_list = [os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_train_mod1.h5ad'),
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_train_mod2.h5ad'),
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_test_mod1.h5ad'),
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_test_mod2.h5ad')]

        self.modalities = []
        for mod_path in mod_path_list:
            self.modalities.append(ad.read_h5ad(mod_path))
        self.loaded = True
        return self

    def sparse_features(self):
        assert self.loaded, 'Data have not been loaded.'
        return [mod.X for mod in self.modalities]

    def numpy_features(self):
        assert self.loaded, 'Data have not been loaded.'
        return [mod.X.toarray() for mod in self.modalities]

class ModalityPredictionDataset(MultiModalityDataset):
    def __init__(self, subtask, data_dir = "./data"):
        assert (subtask in ['openproblems_bmmc_multiome_phase2_mod2', 'openproblems_bmmc_multiome_phase2_rna',
                             'openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_cite_phase2_mod2',
                             'adt2gex', 'gex2adt',
                             'atac2gex', 'gex2atac']), 'Undefined subtask.'

        if subtask == 'adt2gex':
            subtask = 'openproblems_bmmc_cite_phase2_mod2'
        elif subtask == 'gex2adt':
            subtask = 'openproblems_bmmc_cite_phase2_rna'
        elif subtask == 'atac2gex':
            subtask = 'openproblems_bmmc_multiome_phase2_mod2'
        elif subtask == 'gex2atac':
            subtask = 'openproblems_bmmc_multiome_phase2_rna'

        data_url = {'openproblems_bmmc_cite_phase2_mod2': 'https://www.dropbox.com/s/snh8knscnlcq4um/openproblems_bmmc_cite_phase2_mod2.zip?dl=1',
                    'openproblems_bmmc_cite_phase2_rna': 'https://www.dropbox.com/s/xbfyhv830u9pupv/openproblems_bmmc_cite_phase2_rna.zip?dl=1',
                    'openproblems_bmmc_multiome_phase2_mod2': 'https://www.dropbox.com/s/p9ve2ljyy4yqna4/openproblems_bmmc_multiome_phase2_mod2.zip?dl=1',
                    'openproblems_bmmc_multiome_phase2_rna': 'https://www.dropbox.com/s/cz60vp7bwapz0kw/openproblems_bmmc_multiome_phase2_rna.zip?dl=1'}.get(subtask)

        super(ModalityPredictionDataset, self).__init__('predict_modality', data_url, subtask, data_dir)


class ModalityMatchingDataset(MultiModalityDataset):
    def __init__(self, subtask, data_dir = "./data"):
        assert (subtask in ['openproblems_bmmc_multiome_phase2_mod2', 'openproblems_bmmc_multiome_phase2_rna',
                             'openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_cite_phase2_mod2',
                             'adt2gex', 'gex2adt',
                             'atac2gex', 'gex2atac']), 'Undefined subtask.'

        if subtask == 'adt2gex':
            subtask = 'openproblems_bmmc_cite_phase2_mod2'
        elif subtask == 'gex2adt':
            subtask = 'openproblems_bmmc_cite_phase2_rna'
        elif subtask == 'atac2gex':
            subtask = 'openproblems_bmmc_multiome_phase2_mod2'
        elif subtask == 'gex2atac':
            subtask = 'openproblems_bmmc_multiome_phase2_rna'

        data_url = {'openproblems_bmmc_cite_phase2_mod2': 'https://www.dropbox.com/s/fa6zut89xx73itz/openproblems_bmmc_cite_phase2_mod2.zip?dl=1',
                    'openproblems_bmmc_cite_phase2_rna': 'https://www.dropbox.com/s/ep00mqcjmdu0b7v/openproblems_bmmc_cite_phase2_rna.zip?dl=1',
                    'openproblems_bmmc_multiome_phase2_mod2': 'https://www.dropbox.com/s/31qi5sckx768acw/openproblems_bmmc_multiome_phase2_mod2.zip?dl=1',
                    'openproblems_bmmc_multiome_phase2_rna': 'https://www.dropbox.com/s/h1s067wkefs1jh2/openproblems_bmmc_multiome_phase2_rna.zip?dl=1'}.get(subtask)

        super(ModalityMatchingDataset, self).__init__('predict_modality', data_url, subtask, data_dir)

    def load_sol(self):
        assert(self.loaded)
        self.train_sol = ad.read_h5ad(os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_train_sol.h5ad'))
        self.test_sol = ad.read_h5ad(os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_test_sol.h5ad'))
        self.modalities[1] = self.modalities[1][self.train_sol.to_df().values.argmax(1)]
        return self

    def lsi_transform(self, pkl_path=None):

        # TODO: support other two subtasks
        assert self.subtask in ('openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_multiome_phase2_rna'), 'Currently not available.'

        if pkl_path and (not os.path.exists(pkl_path)):

            if self.subtask == 'openproblems_bmmc_cite_phase2_rna':
                lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                m2_train = self.modalities[1].X.toarray()
                m2_test = self.modalities[3].X.toarray()

            if self.subtask == 'openproblems_bmmc_multiome_phase2_rna':
                lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                m2_train = lsi_transformer_atac.fit_transform(self.modalities[1]).values
                m2_test = lsi_transformer_atac.transform(self.modalities[3]).values

            self.transformed_features = [m1_train, m2_train, m1_test, m2_test]
            pickle.dump(self.transformed_features, open(pkl_path, 'wb'))

        else:
            self.transformed_features = pickle.load(open(pkl_path, 'rb'))

        return self

class NIPSJointEmbeddingDataset(MultiModalityDataset):
    def __init__(self, subtask, data_dir = "./data"):
        assert (subtask in ['openproblems_bmmc_multiome_phase2',
                             'openproblems_bmmc_cite_phase2',
                             'adt', 'atac']), 'Undefined subtask.'

        if subtask == 'adt':
            subtask = 'openproblems_bmmc_cite_phase2'
        elif subtask == 'atac':
            subtask = 'openproblems_bmmc_multiome_phase2'

        data_url = {'openproblems_bmmc_cite_phase2': 'https://www.dropbox.com/s/fa6zut89xx73itz/openproblems_bmmc_cite_phase2_mod2.zip?dl=1',
                    'openproblems_bmmc_multiome_phase2': 'https://www.dropbox.com/s/31qi5sckx768acw/openproblems_bmmc_multiome_phase2_mod2.zip?dl=1'}
        super(ModalityMatchingDataset, self).__init__('predict_modality', data_url, subtask, data_dir)

    def load_metadata(self):
        assert(self.loaded)
        # TODO exploration loading
        return self

    def lsi_transform(self, pkl_path=None):

        assert self.subtask in ('openproblems_bmmc_cite_phase2', 'openproblems_bmmc_multiome_phase2')

        if pkl_path and (not os.path.exists(pkl_path)):

            if self.subtask == 'openproblems_bmmc_cite_phase2':
                lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                m2_train = self.modalities[1].X.toarray()
                m2_test = self.modalities[3].X.toarray()

            if self.subtask == 'openproblems_bmmc_multiome_phase2':
                lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                m2_train = lsi_transformer_atac.fit_transform(self.modalities[1]).values
                m2_test = lsi_transformer_atac.transform(self.modalities[3]).values

            self.transformed_features = [m1_train, m2_train, m1_test, m2_test]
            pickle.dump(self.transformed_features, open(pkl_path, 'wb'))

        else:
            self.transformed_features = pickle.load(open(pkl_path, 'rb'))

        return self

class NIPSJointEmbeddingDataset(MultiModalityDataset):
    def __init__(self, subtask, data_dir = "./data"):
        assert (subtask in ['openproblems_bmmc_multiome_phase2',
                             'openproblems_bmmc_cite_phase2',
                             'adt', 'atac']), 'Undefined subtask.'

        if subtask == 'adt':
            subtask = 'openproblems_bmmc_cite_phase2'
        elif subtask == 'atac':
            subtask = 'openproblems_bmmc_multiome_phase2'

        data_url = {'openproblems_bmmc_cite_phase2': 'https://www.dropbox.com/s/fa6zut89xx73itz/openproblems_bmmc_cite_phase2_mod2.zip?dl=1',
                    'openproblems_bmmc_multiome_phase2': 'https://www.dropbox.com/s/31qi5sckx768acw/openproblems_bmmc_multiome_phase2_mod2.zip?dl=1'}
        super(ModalityMatchingDataset, self).__init__('predict_modality', data_url, subtask, data_dir)

    def load_metadata(self):
        assert(self.loaded)
        # TODO exploration loading
        return self

    def lsi_transform(self, pkl_path=None):

        assert self.subtask in ('openproblems_bmmc_cite_phase2', 'openproblems_bmmc_multiome_phase2')

        if pkl_path and (not os.path.exists(pkl_path)):

            if self.subtask == 'openproblems_bmmc_cite_phase2':
                lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                m2_train = self.modalities[1].X.toarray()
                m2_test = self.modalities[3].X.toarray()

            if self.subtask == 'openproblems_bmmc_multiome_phase2':
                lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                m2_train = lsi_transformer_atac.fit_transform(self.modalities[1]).values
                m2_test = lsi_transformer_atac.transform(self.modalities[3]).values

            self.transformed_features = [m1_train, m2_train, m1_test, m2_test]
            pickle.dump(self.transformed_features, open(pkl_path, 'wb'))

        else:
            self.transformed_features = pickle.load(open(pkl_path, 'rb'))

        return self