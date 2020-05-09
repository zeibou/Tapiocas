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
            enabled_str = f" - OFF"
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
        blur = cv2.GaussianBlur(image, area, 0)
        return blur


class CannyFilter(Filter):
    def __init__(self):
        super().__init__(.33)

    @staticmethod
    def name():
        return "Canny"

    def set_value(self, v):
        v = float(v)
        if 0 <= v <= 1:
            super().set_value(v)

    def apply(self, image):
        median = np.median(image)
        lower = int(max(0, (1.0 - self.value) * median))
        upper = int(min(255, (1.0 + self.value) * median))
        canny_output = cv2.Canny(image, lower, upper)
        image = cv2.cvtColor(canny_output, cv2.COLOR_GRAY2BGR)
        return image


class ContourFilter(Filter):
    def __init__(self):
        super().__init__(1)

    @staticmethod
    def name():
        return "Draw Contours"

    def set_value(self, v):
        v = int(v)
        if 0 < v <= 50:
            super().set_value(v)

    def apply(self, image):
        # Find contours
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours
        drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        color = (255, 255, 255)
        cv2.drawContours(drawing, contours, -1, color, self.value, cv2.LINE_AA)
        return drawing


class InvertFilter(Filter):
    @staticmethod
    def name():
        return "Invert Colors"

    def set_value(self, v):
        # no value needed, we keep None
        pass

    def apply(self, image):
        return cv2.bitwise_not(image)


class ThresholdFilter(Filter):
    @staticmethod
    def name():
        return "Adaptive Threshold"

    def set_value(self, v):
        pass

    def apply(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ret = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        return ret
