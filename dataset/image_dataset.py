from .dataset import DataSet

import math
from typing import Union

import cv2
import imagesize
import json
import tqdm


class ImageDataSet(DataSet):
    def create_valid_indices(self, pairs: list):
        valid_indices = {}

        description_prefix = "Checking validity: "

        tqdm_iterator = tqdm.tqdm(pairs, desc=description_prefix)
        for index, (image_file_path, annotation_file_path) in enumerate(tqdm_iterator):
            tqdm_iterator.set_description(description_prefix + image_file_path)

            label = self.extract_label(image_file_path)

            image_width, image_height = imagesize.get(image_file_path)

            annotation = self.parse_annotation(annotation_file_path)

            if self.is_valid_annotation(image_width, image_height, annotation) is False:
                continue

            if label not in valid_indices.keys():
                valid_indices[label] = []

            valid_indices[label].append(index)

        return valid_indices

    def train_count(self, label):
        return len(self.train_valid_indices[label])

    def validation_count(self, label):
        return len(self.validation_valid_indices[label])

    def get_train_filename(self, label: Union[int, str], index: int):
        pair_index = self.train_valid_indices[label][index]
        image_file_path, annotation_file_path = self.train_pairs[pair_index]

        return image_file_path, annotation_file_path

    def get_validation_filename(self, label: Union[int, str], index: int):
        pair_index = self.validation_valid_indices[label][index]
        image_file_path, annotation_file_path = self.validation_pairs[pair_index]

        return image_file_path, annotation_file_path

    def get_train_datum(self, label, index):
        pair_index = self.train_valid_indices[label][index]

        image_file_path, annotation_file_path = self.train_pairs[pair_index]

        annotation = self.parse_annotation(annotation_file_path)

        image = cv2.imread(image_file_path)

        return image, annotation

    def get_validation_datum(self, label, index):
        pair_index = self.validation_valid_indices[label][index]

        image_file_path, annotation_file_path = self.validation_pairs[pair_index]

        annotation = self.parse_annotation(annotation_file_path)

        image = cv2.imread(image_file_path)

        return image, annotation
