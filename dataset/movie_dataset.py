from .dataset import DataSet

import math
import random

import cv2
import json
import tqdm


class MovieDataSet(DataSet):
    random_salt = 20200305

    def set_random_salt(self, random_salt):
        self.random_salt = random_salt

    @staticmethod
    def create_valid_indices(pairs: list, label_extraction_function):
        rotating_margin_factor = (math.sqrt(2.0) - 1.0) / 2.0

        valid_indices = {}

        description_prefix = "Checking validity: "

        tqdm_iterator = tqdm.tqdm(pairs, desc=description_prefix)
        for video_index, (video_file_path, json_file_path) in enumerate(tqdm_iterator):
            tqdm_iterator.set_description(description_prefix + video_file_path)

            label = label_extraction_function(video_file_path)

            video = cv2.VideoCapture(video_file_path)

            video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

            valid_frame_indices = []

            with open(json_file_path) as file:
                annotations = json.load(file)

                for frame_index, annotation in enumerate(annotations):
                    if "landmark" not in annotation.keys():
                        continue

                    if len(annotation['landmark']) != 106:
                        continue

                    face_box = DataSet.make_box_from_landmark(annotation['landmark'])
                    margin_need = face_box[2] * rotating_margin_factor

                    x = int(round(face_box[0] - margin_need))
                    y = int(round(face_box[1] - margin_need))
                    width = int(round(face_box[2] + (2.0 * margin_need)))
                    height = int(round(face_box[3] + (2.0 * margin_need)))

                    if x < 0 or y < 0:
                        continue

                    if x + width >= video_width or y + height >= video_height:
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

    def get_train_datum(self, label, index):
        video_indices = sorted(list(self.train_valid_indices[label].keys()))
        video_index = video_indices[index]

        random.seed(None)
        frame_index = random.choice(self.train_valid_indices[label][video_index])

        movie_file_path, json_file_path = self.train_pairs[video_index]

        with open(json_file_path) as json_file:
            annotations = json.load(json_file)
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

        movie_file_path, json_file_path = self.validation_pairs[video_index]

        with open(json_file_path) as json_file:
            annotations = json.load(json_file)
            annotation = annotations[frame_index]

        video = cv2.VideoCapture(movie_file_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        _, image = video.read()

        return image, annotation
