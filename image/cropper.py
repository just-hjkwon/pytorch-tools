import numpy

class Cropper:
    @staticmethod
    def crop(image: numpy.ndarray, x: int, y: int, width: int, height: int):
        crop_image_shape = list(image.shape)
        crop_image_shape[0] = width
        crop_image_shape[1] = height

        crop_image = numpy.zeros(crop_image_shape, image.dtype)

        copy_width = width
        copy_height = height

        if x >= 0:
            source_x = x
            target_x = 0
        else:
            source_x = 0
            target_x = -x
            copy_width = width + x

        if y >= 0:
            source_y = y
            target_y = 0
        else:
            source_y = 0
            target_y = -y
            copy_height = height + x

        if source_x + copy_width >= image.shape[1]:
            copy_width = image.shape[1] - source_x

        if source_y + copy_height >= image.shape[0]:
            copy_height = image.shape[0] - source_y

        crop_image[target_y:target_y+copy_height, target_x:target_x+copy_width, :] = image[source_y:source_y+copy_height, source_x:source_x+copy_width, :]

        return crop_image