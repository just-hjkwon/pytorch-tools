from .dataset import DataSet

import math

import cv2
import imagesize
import json
import tqdm


class ImageDataSet(DataSet):
    @staticmethod
    def create_valid_indices(pairs: list, label_extraction_function):
        rotating_margin_factor = (math.sqrt(2.0) - 1.0) / 2.0

        valid_indices = {}

        description_prefix = "Checking validity: "

        tqdm_iterator = tqdm.tqdm(pairs, desc=description_prefix)
        for index, (image_file_path, json_file_path) in enumerate(tqdm_iterator):
            tqdm_iterator.set_description(description_prefix + image_file_path)

            label = label_extraction_function(image_file_path)

            image_width, image_height = imagesize.get(image_file_path)

            with open(json_file_path) as file:
                annotation = json.load(file)

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

                if x + width >= image_width or y + height >= image_height:
                    continue

                if label not in valid_indices.keys():
                    valid_indices[label] = []

                valid_indices[label].append(index)

        return valid_indices

    def train_count(self, label):
        return len(self.train_valid_indices[label])

    def validation_count(self, label):
        return len(self.validation_valid_indices[label])

    def get_train_datum(self, label, index):
        pair_index = self.train_valid_indices[label][index]

        image_file_path, json_file_path = self.train_pairs[pair_index]

        with open(json_file_path) as json_file:
            annotation = json.load(json_file)

        image = cv2.imread(image_file_path)

        return image, annotation

    def get_validation_datum(self, label, index):
        pair_index = self.validation_valid_indices[label][index]

        image_file_path, json_file_path = self.validation_pairs[pair_index]

        with open(json_file_path) as json_file:
            annotation = json.load(json_file)

        image = cv2.imread(image_file_path)

        return image, annotation
