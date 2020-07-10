from .dataset import DataSet

import math
import random
from typing import Union

import cv2
import tqdm


class MovieDataSet(DataSet):
    def create_valid_indices(self, pairs: list):
        valid_indices = {}

        description_prefix = "Checking validity: "

        tqdm_iterator = tqdm.tqdm(pairs, desc=description_prefix)
        for video_index, (video_file_path, annotation_file_path) in enumerate(tqdm_iterator):
            tqdm_iterator.set_description(description_prefix + video_file_path)

            label = self.extract_label(video_file_path)

            video = cv2.VideoCapture(video_file_path)

            video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

            valid_frame_indices = []

            annotations = self.parse_annotation(annotation_file_path)

            for frame_index, annotation in enumerate(annotations):
                if self.is_valid_annotation(video_width, video_height, annotation) is False:
                    continue

                valid_frame_indices.append(frame_index)

            if len(valid_frame_indices) == 0:
                continue

            if label not in valid_indices.keys():
                valid_indices[label] = {}

            valid_indices[label][video_index] = valid_frame_indices

        return valid_indices

    def train_count(self, label):
        return len(self.train_valid_indices[label].keys())

    def validation_count(self, label):
        return len(self.validation_valid_indices[label].keys())

    def get_train_filename(self, label: Union[int, str], index: int):
        video_indices = sorted(list(self.train_valid_indices[label].keys()))
        video_index = video_indices[index]

        movie_file_path, annotation_file_path = self.train_pairs[video_index]

        return movie_file_path, annotation_file_path

    def get_validation_filename(self, label: Union[int, str], index: int):
        video_indices = sorted(list(self.validation_valid_indices[label].keys()))
        video_index = video_indices[index]

        movie_file_path, annotation_file_path = self.validation_pairs[video_index]

        return movie_file_path, annotation_file_path

    def get_train_datum(self, label, index):
        video_indices = sorted(list(self.train_valid_indices[label].keys()))
        video_index = video_indices[index]

        random.seed(None)
        frame_index = random.choice(self.train_valid_indices[label][video_index])

        movie_file_path, annotation_file_path = self.train_pairs[video_index]

        annotations = self.parse_annotation(annotation_file_path)
        annotation = annotations[frame_index]

        video = cv2.VideoCapture(movie_file_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        _, image = video.read()

        return image, annotation

    def get_validation_datum(self, label, index):
        video_indices = sorted(list(self.validation_valid_indices[label].keys()))
        video_index = video_indices[index]

        random.seed(self.random_salt + index)
        frame_index = random.choice(self.validation_valid_indices[label][video_index])

        movie_file_path, annotation_file_path = self.validation_pairs[video_index]

        annotations = self.parse_annotation(annotation_file_path)
        annotation = annotations[frame_index]

        video = cv2.VideoCapture(movie_file_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        _, image = video.read()

        return image, annotation
