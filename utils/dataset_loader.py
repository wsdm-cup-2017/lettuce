# -*- coding: utf-8 -*-

import os
from collections import defaultdict


class DatasetLoader(object):
    def __init__(self, dataset_dir):
        self.nationality_kb = self._load_kb_file(os.path.join(dataset_dir, 'nationality.kb'))
        self.nationality_train = self._load_train_file(os.path.join(dataset_dir, 'nationality.train'))
        self.profession_kb = self._load_kb_file(os.path.join(dataset_dir, 'profession.kb'))
        self.profession_train = self._load_train_file(os.path.join(dataset_dir, 'profession.train'))

    @staticmethod
    def _load_kb_file(kb_file):
        ret = defaultdict(list)
        with open(kb_file) as f:
            for line in f:
                (title, type_name) = line.rstrip().decode('utf-8').split('\t')
                ret[title].append(type_name)

        return ret.items()

    @staticmethod
    def _load_train_file(train_file):
        ret = defaultdict(list)
        with open(train_file) as f:
            for line in f:
                (title, type_name, score) = line.rstrip().decode('utf-8').split('\t')
                ret[title].append((type_name, int(score)))

        return ret.items()
