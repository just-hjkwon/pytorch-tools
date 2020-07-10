from abc import ABC, abstractmethod
import glob
import os
from typing import Union, List

import numpy as np


class DataSet(ABC):
    def __init__(self, base_directory: str, target_extension: Union[str, List[str]], anntoation_extension: Union[None, str]=None):
        self.base_directory = base_directory

        if type(target_extension) == list:
            self.target_extension = target_extension
        elif type(target_extension) == str:
            self.target_extension = [target_extension]
        else:
            raise ValueError("The target_extension should be the string of extension or list of it.")

        if anntoation_extension is not None:
            self.anntoation_extension = anntoation_extension
        else:
            self.anntoation_extension = "json"

        self.train_pairs = []
        self.validation_pairs = []

        self.prepare_pairs()

        self.train_valid_indices = self.create_valid_indices(self.train_pairs)
        self.validation_valid_indices = self.create_valid_indices(self.validation_pairs)

        self.labels = set(self.train_valid_indices.keys()) | set(self.validation_valid_indices.keys())

        self.is_train_mode = True
        self.random_salt = 20200305

    def prepare_pairs(self):
        file_list = []
        for extension in self.target_extension:
            file_list += glob.glob(self.base_directory + "/**/*.%s" % extension, recursive=True)

        train_file_list = list(filter(self.train_datum_filter, file_list))
        validation_file_list = list(filter(self.validation_datum_filter, file_list))

        train_file_list.sort()
        validation_file_list.sort()

        self.train_pairs = DataSet.create_pairs_with_annotation(train_file_list, self.anntoation_extension)
        self.validation_pairs = DataSet.create_pairs_with_annotation(validation_file_list, self.anntoation_extension)

    def set_train_mode(self):
        self.is_train_mode = True

    def set_validation_mode(self):
        self.is_train_mode = False

    def count(self, label):
        if self.is_train_mode is True:
            return self.train_count(label)
        else:
            return self.validation_count(label)

    def get_labels(self):
        return self.labels

    def get_filename(self, label, index):
        if self.is_train_mode is True:
            return self.get_train_filename(label, index)
        else:
            return self.get_validation_filename(label, index)

    def get_datum(self, label, index):
        if self.is_train_mode is True:
            return self.get_train_datum(label, index)
        else:
            return self.get_validation_datum(label, index)

    def set_random_salt(self, random_salt):
        self.random_salt = 20200305 + random_salt

    @staticmethod
    def create_pairs_with_annotation(file_list: list, annotatino_extension: str):
        pairs = []

        for file_path in file_list:
            annotation_file_path = os.path.splitext(file_path)[0] + ".%s" % annotatino_extension

            if os.path.exists(annotation_file_path) is True:
                pairs.append((file_path, annotation_file_path))
            else:
                continue

        return pairs

    @abstractmethod
    def get_train_filename(self, label: Union[int, str], index: int):
        pass

    @abstractmethod
    def get_validation_filename(self, label: Union[int, str], index: int):
        pass

    @abstractmethod
    def train_count(self, label):
        pass

    @abstractmethod
    def validation_count(self, label):
        pass

    @abstractmethod
    def get_train_datum(self, label, index):
        pass

    @abstractmethod
    def get_validation_datum(self, label, index):
        pass

    @abstractmethod
    def create_valid_indices(self, file_list: list):
        pass

    @staticmethod
    @abstractmethod
    def train_datum_filter(file_path: str):
        pass

    @staticmethod
    @abstractmethod
    def validation_datum_filter(file_path: str):
        pass

    @staticmethod
    @abstractmethod
    def extract_label(file_path: str):
        pass

    @staticmethod
    @abstractmethod
    def parse_annotation(file_path: str):
        pass

    @staticmethod
    @abstractmethod
    def is_valid_annotation(image_width, image_height, annotation):
        pass
