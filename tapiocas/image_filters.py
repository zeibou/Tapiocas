import abc
import numpy as np
import cv2


class Filter(metaclass=abc.ABCMeta):
    def __init__(self, value=None):
        self.value = value
        self.enabled = True

    @staticmethod
    @abc.abstractmethod
    def name():
        pass

    @abc.abstractmethod
    def set_value(self, v):
        self.value = v

    @abc.abstractmethod
    def apply(self, image):
        pass

    def __str__(self):
        value_str = ""
        enabled_str = ""
        if self.value is not None:
            value_str = f" ({self.value})"
        if not self.enabled:
            enabled_str = f" - off"
        return f"{self.name()}{value_str}{enabled_str}"


class GrayFilter(Filter):
    @staticmethod
    def name():
        return "Gray"

    def set_value(self, v):
        # no value needed, we keep None
        pass

    def apply(self, image):
        # convert to gray (matrix shape will change to a single channel)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # converts back to RGB (pixels will stay gray) so that we can draw in color again
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray


class BlurFilter(Filter):
    def __init__(self):
        super().__init__(3)

    @staticmethod
    def name():
        return "Blur"

    def set_value(self, v):
        v = int(v)
        if 0 < v < 100:
            super().set_value(v)

    def apply(self, image):
        area = (self.value, self.value)
        blur = cv2.blur(image, area)
        return blur


class CannyContourFilter(Filter):
    def __init__(self):
        super().__init__(100)

    @staticmethod
    def name():
        return "CannyContour"

    def set_value(self, v):
        v = int(v)
        if 0 <= v < 256:
            super().set_value(v)

    def apply(self, image):
        canny_output = cv2.Canny(image, self.value, self.value * 2)
        # Find contours
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (255, 0, 0)
            cv2.drawContours(drawing, contours, i, color, 1, cv2.LINE_4, hierarchy, 0)
        return drawing

