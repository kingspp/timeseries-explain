import hashlib
import json
import datetime
from nte.utils import get_md5_checksum
from nte.utils.plot_utils import intialize_plot
import random
import numpy as np
import torch
import torch.utils.data as tdata
from fastdtw import fastdtw
from sklearn.metrics import euclidean_distances
import os
from abc import ABCMeta, abstractmethod
from nte.utils import CustomJsonEncoder


class Dataset(tdata.Dataset, metaclass=ABCMeta):
    def __init__(self, name, meta_file_path, bb_model=None):
        print("Loading train data . . .")
        self.train_data, self.train_label = self.load_train_data()
        print("Loading test data . . .")
        self.test_data, self.test_label = self.load_test_data()
        self.name = name
        self.meta_file_path = meta_file_path
        self.bb_model = bb_model

        try:
            print("Loading meta file . . .")
            self.meta = json.load(open(meta_file_path))
            # Load Train Summary Statistics
            self.train_class_0_indices = self.meta['train_class_0_indices']
            self.train_class_1_indices = self.meta['train_class_1_indices']
            self.train_class_0_data = np.take(self.train_data, self.train_class_0_indices, axis=0)
            self.train_class_1_data = np.take(self.train_data, self.train_class_1_indices, axis=0)
            self.train_class_0_mean = np.array(self.meta['train_class_0_mean'])
            self.train_class_1_mean = np.array(self.meta['train_class_1_mean'])

            # Load Test Summary Statistics
            self.test_class_0_indices = self.meta['test_class_0_indices']
            self.test_class_1_indices = self.meta['test_class_1_indices']
            self.test_class_0_data = np.take(self.test_data, self.test_class_0_indices, axis=0)
            self.test_class_1_data = np.take(self.test_data, self.test_class_1_indices, axis=0)
            self.test_class_0_mean = np.array(self.meta['test_class_0_mean'])
            self.test_class_1_mean = np.array(self.meta['test_class_1_mean'])
            self.train_statistics = self.meta['train_statistics']
            self.test_statistics = self.meta['test_statistics']
            print("Meta file loaded successfully")
        except Exception as e:
            print("Meta file not found. Creating meta file . . .")
            # Generate Train Summary Statistics
            self.train_class_0_indices = np.where(self.train_label == 0)[0]
            self.train_class_1_indices = np.where(self.train_label == 1)[0]
            self.train_class_0_data = np.take(self.train_data, self.train_class_0_indices, axis=0)
            self.train_class_1_data = np.take(self.train_data, self.train_class_1_indices, axis=0)
            self.train_class_0_mean = self.train_class_0_data.mean(axis=0)
            self.train_class_1_mean = self.train_class_1_data.mean(axis=0)

            # Generate Test Summary Statistics
            self.test_class_0_indices = np.where(self.test_label == 0)[0]
            self.test_class_1_indices = np.where(self.test_label == 1)[0]
            self.test_class_0_data = np.take(self.test_data, self.test_class_0_indices, axis=0)
            self.test_class_1_data = np.take(self.test_data, self.test_class_1_indices, axis=0)
            self.test_class_0_mean = self.test_class_0_data.mean(axis=0)
            self.test_class_1_mean = self.test_class_1_data.mean(axis=0)
            self.train_statistics = self.sample(dist_typ='dtw', data='train')
            self.test_statistics = self.sample(dist_typ='dtw', data='test')
            self.meta = {
                'train_class_0_indices': self.train_class_0_indices,
                'train_class_1_indices': self.train_class_1_indices,
                'train_class_0_mean': self.train_class_0_mean,
                'train_class_1_mean': self.train_class_1_mean,
                'test_class_0_indices': self.test_class_0_indices,
                'test_class_1_indices': self.test_class_1_indices,
                'test_class_0_mean': self.test_class_0_mean,
                'test_class_1_mean': self.test_class_1_mean,
                'train_statistics': self.train_statistics,
                'test_statistics': self.test_statistics
            }
            with open(self.meta_file_path, 'w') as f:
                json.dump(self.meta, f, indent=2, cls=CustomJsonEncoder)
            print("Meta file created successfully")
        self.valid_data = []
        self.valid_label = []

        # self.train_meta = None
        # self.test_meta = None
        # self.valid_meta = None
        self.indices = {}

        # Prepare representatives
        self.representatives = self.representatives = {'train': self._generate_representatives('train'),
                                                       'test': self._generate_representatives('test')}
        self.valid_name = list(self.representatives['train'].keys())
        self.valid_data = np.array(self.valid_data)
        self.valid_label = np.array(self.valid_label)

    def _generate_representatives(self, data_type="train"):
        representatives = {}
        for k, v in self.meta[data_type + '_statistics']['between_class'].items():
            for e, vals in enumerate(v[:2]):
                representatives[f"between_class_{k}_class_{e}"] = vals
                if data_type == 'test':
                    self.valid_data.append(vals)
                    self.valid_label.append(float(e))

        for k, v in self.meta[data_type + '_statistics']['among_class_a'].items():
            for e, vals in enumerate(v[:2]):
                representatives[f"among_class_0_{k}_sample_{e}"] = vals
                if data_type == 'test':
                    self.valid_data.append(vals)
                    self.valid_label.append(0.0)

        for k, v in self.meta[data_type + '_statistics']['among_class_b'].items():
            for e, vals in enumerate(v[:2]):
                representatives[f"among_class_1_{k}_sample_{e}"] = vals
                if data_type == 'test':
                    self.valid_data.append(vals)
                    self.valid_label.append(1.0)

        for k, v in self.meta[data_type + '_statistics']['percentiles_a_data'].items():
            representatives[f"class_0_percentile_{k}"] = v[0]
            if data_type == 'test':
                self.valid_data.append(v[0])
                self.valid_label.append(0.0)

        for k, v in self.meta[data_type + '_statistics']['percentiles_b_data'].items():
            representatives[f"class_1_percentile_{k}"] = v[0]
            if data_type == 'test':
                self.valid_data.append(v[0])
                self.valid_label.append(1.0)
        return representatives

    def describe(self):
        stats = {
            'Timeseries Length': self.train_data.shape[1],
            'Train Samples': self.train_data.shape[0],
            'Test Samples': self.test_data.shape[0],
            'Train Event Rate': np.mean(self.train_label),
            'Test Event Rate': np.mean(self.test_label)
        }
        print(stats)
        return stats

    @abstractmethod
    def load_train_data(self) -> ():
        pass

    @abstractmethod
    def load_test_data(self) -> ():
        pass

    def _create_valid_from_train(self, valid_ratio: 0.2):
        val_index = self.train_data.shape[0] - int(self.train_data.shape[0] * valid_ratio)
        self.train_data, self.train_label = self.train_data[:val_index], self.train_label[:val_index]
        self.valid_data, self.valid_label = self.train_data[val_index:], self.train_label[val_index:]

    def __getitem__(self, ix):
        return self.train_data[ix], self.train_label[ix]

    def __len__(self):
        return len(self.train_data)

    def get_random_sample(self, cls=None):
        # todo Validation
        if cls is None:
            r = random.randrange(0, len(self.train_data))
            return self.train_data[r], self.train_label[r], r
        else:
            if cls not in self.indices:
                self.indices[cls] = np.where(self.train_label == cls)[0]
            r = random.randrange(0, len(self.indices[cls]))
            return self.train_data[self.indices[cls][r]], self.train_label[self.indices[cls][r]], self.indices[cls][
                r]

    def read_meta(self, file_list):
        meta_data = []
        for file in file_list:
            meta_data.append(json.load(open(file)))
        return meta_data

    def create_meta(self, file_list):
        md5checksum = get_md5_checksum(file_list)
        for file, md5 in zip(file_list, md5checksum):
            with open(file.split('.')[0] + '.meta', 'w') as f:
                json.dump({"md5": md5, "timestamp": str(datetime.datetime.now())}, f)

    def batch(self, batch_size=32):
        """
        Function to batch the data
        :param batch_size: batches
        :return: batches of X and Y
        """
        l = len(self.train_data)
        for ndx in range(0, l, batch_size):
            yield self.train_data[ndx:min(ndx + batch_size, l)], self.train_label[ndx:min(ndx + batch_size, l)]

    def sample(self, dist_typ='dtw', data='test', percentiles=[0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]):

        print(f"Computing representative samples for {data} using {dist_typ} distance")
        if dist_typ == 'euc':
            dist_fn = lambda a, b: np.linalg.norm(a - b)
        elif dist_typ == 'dtw':
            dist_fn = lambda a, b: fastdtw(a, b)[0]

        if data == 'test':
            class_0_data = self.test_class_0_data
            class_1_data = self.test_class_1_data
            class_0_indices = self.test_class_0_indices
            class_1_indices = self.test_class_1_indices
        elif data == 'train':
            class_0_data = self.train_class_0_data
            class_1_data = self.train_class_1_data
            class_0_indices = self.train_class_0_indices
            class_1_indices = self.train_class_1_indices

        # Between Classes - Max 2, Min 2
        # Among Classes - Max 2, Min 2

        samples = {'between_class': {'opposing': [], 'similar': []},
                   'among_class_a': {'opposing': [], 'similar': []},
                   'among_class_b': {'opposing': [], 'similar': []},
                   'percentiles_a': [], 'percentiles_b': []}
        # Get opposing classes
        min_dist, max_dist = float('inf'), 0.0

        # Between Class
        print("Computing between class samples . . .")
        for ea, point_a in enumerate(class_0_data):
            for eb, point_b in enumerate(class_1_data):
                dist = dist_fn(point_a.reshape([1, -1]), point_b.reshape([1, -1]))
                if dist < min_dist:
                    min_dist = dist
                    samples['between_class']['similar'] = (
                        point_a, point_b, min_dist, [class_0_indices[ea], class_1_indices[eb]])
                if dist > max_dist:
                    max_dist = dist
                    samples['between_class']['opposing'] = (
                        point_a, point_b, max_dist, [class_0_indices[ea], class_1_indices[eb]])

        # Among Class
        print("Computing among class 0 samples . . .")
        min_dist, max_dist = float('inf'), 0.0
        for ea, point_a in enumerate(class_0_data):
            for eb, point_b in enumerate(class_0_data):
                if ea != eb:
                    dist = dist_fn(point_a.reshape([1, -1]), point_b.reshape([1, -1]))
                    if dist < min_dist:
                        min_dist = dist
                        samples['among_class_a']['similar'] = (
                            point_a, point_b, min_dist, [class_0_indices[ea], class_0_data[eb]])
                    if dist > max_dist:
                        max_dist = dist
                        samples['among_class_a']['opposing'] = (
                            point_a, point_b, max_dist, [class_0_indices[ea], class_0_data[eb]])

        print("Computing among class 1 samples . . .")
        min_dist, max_dist = float('inf'), 0.0
        for ea, point_a in enumerate(class_1_data):
            for eb, point_b in enumerate(class_1_data):
                if ea != eb:
                    dist = dist_fn(point_a.reshape([1, -1]), point_b.reshape([1, -1]))
                    if dist < min_dist:
                        min_dist = dist
                        samples['among_class_b']['similar'] = (
                            point_a, point_b, min_dist, [class_1_indices[ea], class_1_indices[eb]])
                    if dist > max_dist:
                        max_dist = dist
                        samples['among_class_b']['opposing'] = (
                            point_a, point_b, max_dist, [class_1_indices[ea], class_1_indices[eb]])

        print("Computing percentiles . . .")
        # Get percentiles for each classes
        samples['percentiles_a'] = {percentiles[e]: q for e, q in
                                    enumerate(np.quantile(class_0_data, q=percentiles, axis=0))}
        samples['percentiles_b'] = {percentiles[e]: q for e, q in
                                    enumerate(np.quantile(class_1_data, q=percentiles, axis=0))}

        samples['percentiles_a_data'] = {q: [] for q, _ in samples['percentiles_a'].items()}
        samples['percentiles_b_data'] = {q: [] for q, _ in samples['percentiles_b'].items()}

        print("Matching percentiles for class 0 . . .")
        for q, percentile in samples['percentiles_a'].items():
            min_dist = float('inf')
            for point_a in class_0_data:
                dist = dist_fn(point_a, percentile)
                if dist < min_dist:
                    samples['percentiles_a_data'][q] = (point_a, dist)

        print("Matching percentiles for class 1 . . .")
        for q, percentile in samples['percentiles_b'].items():
            min_dist = float('inf')
            for point_b in class_1_data:
                dist = dist_fn(point_b, percentile)
                if dist < min_dist:
                    samples['percentiles_b_data'][q] = (point_b, dist)

        return samples

    def visualize(self, display=True):
        plt = intialize_plot()
        plt.figure(figsize=(15, 10))
        for i in range(5):
            ax = plt.subplot(int(f"32{i + 1}"), sharex=ax if i > 0 else None)
            d, l, idx = self.get_random_sample(cls=0)
            plt.plot(d, label=f"Idx: {idx} Class {l}")
            d, l, idx = self.get_random_sample(cls=1)
            plt.plot(d, label=f"Idx: {idx} Class {l}")
            plt.legend()
        plt.xlabel("Timesteps")
        plt.subplot(326)
        d = self.train_data[np.where(self.train_label == 0)].mean(0)
        plt.plot(d, label=f"Mean of Class {0}")
        d = self.train_data[np.where(self.train_label == 1)].mean(0)
        plt.plot(d, label=f"Mean of Class {1}")
        plt.xlabel("Timesteps")
        plt.legend()
        plt.suptitle(f"Summary of {self.name}", fontsize=18)
        if display:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
