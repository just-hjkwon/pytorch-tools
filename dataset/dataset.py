from abc import ABC, abstractmethod
import glob
import os

import numpy as np


class DataSet(ABC):
    def __init__(self, base_directory: str, target_extension: str):
        self.base_directory = base_directory
        self.target_extension = target_extension

        self.train_pairs = []
        self.validation_pairs = []

        self.prepare_pairs()

        self.train_valid_indices = self.create_valid_indices(self.train_pairs, self.label_extraction_function)
        self.validation_valid_indices = self.create_valid_indices(self.validation_pairs, self.label_extraction_function)

        self.labels = set(self.train_valid_indices.keys()) | set(self.validation_valid_indices.keys())

        self.is_train_mode = True
        self.random_salt = 20200305

    def prepare_pairs(self):
        file_list = glob.glob(self.base_directory + "/**/*.%s" % self.target_extension, recursive=True)

        train_file_list = list(filter(self.train_datum_filter, file_list))
        validation_file_list = list(filter(self.validation_datum_filter, file_list))

        train_file_list.sort()
        validation_file_list.sort()

        self.train_pairs = DataSet.create_pairs_with_json(train_file_list)
        self.validation_pairs = DataSet.create_pairs_with_json(validation_file_list)

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

    def get_datum(self, label, index):
        if self.is_train_mode is True:
            return self.get_train_datum(label, index)
        else:
            return self.get_validation_datum(label, index)

    def set_random_salt(self, random_salt):
        self.random_salt = 20200305 + random_salt

    @staticmethod
    def create_pairs_with_json(file_list: list):
        pairs = []

        for file_path in file_list:
            json_file_path = os.path.splitext(file_path)[0] + ".json"

            if os.path.exists(json_file_path) is True:
                pairs.append((file_path, json_file_path))
            else:
                continue

        return pairs

    @staticmethod
    def make_box_from_landmark(landmark: list):
        landmark = [[ls["x"], ls["y"]] for ls in landmark]
        landmark_array = np.array(landmark)

        min_xy = landmark_array.min(axis=0)
        max_xy = landmark_array.max(axis=0)

        face_box = [min_xy[0], min_xy[1], max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]]

        face_box[1] -= face_box[3] * 0.1
        face_box[3] *= 1.15

        if face_box[2] > face_box[3]:
            padding = (face_box[2] - face_box[3]) / 2.0
            face_box[1] -= padding
            face_box[3] = face_box[2]
        else:
            padding = (face_box[3] - face_box[2]) / 2.0
            face_box[0] -= padding
            face_box[2] = face_box[3]

        return face_box

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

    @staticmethod
    @abstractmethod
    def create_valid_indices(file_list: list, label_extraction_function):
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
    def label_extraction_function(file_path: str):
        pass
