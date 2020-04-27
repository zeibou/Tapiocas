import io
import PySimpleGUI as sg
from PIL import Image
import logging
import collections
import threading
import time
import random

from adb_connector import AdbConnector
import config_manager
import log_manager

KEY_BUTTON_SCREENSHOT = "SCREENSHOT-key"
KEY_BUTTON_IMG_CLICK = "MAIN_IMAGE_CLICK-key"
KEY_BUTTON_ZOOM_FRM_CLOSE = "ZOOM_IMAGE_CLOSE-key"
KEY_SLIDER_ZOOM = "ZOOM_IMAGE_SLIDER-key"

DISPLAY_MAX_SIZE = (300, 600)
SCREENSHOT_RAW = False
SCREENSHOT_PULL = False

ZOOM_RADIUS_MIN = 10
ZOOM_RADIUS_MAX = 100
START_COLOR = (random.randint(25, 100), random.randint(25, 100), random.randint(25, 100))


class Model:
    device_screen_size: (int, int)
    zoom_center: (int, int)
    zoom_radius: int

    main_image: Image
    zoom_image = Image

    main_image_element: sg.Image
    zoom_image_element: sg.Image
    zoom_column: sg.Column
    coord_label_element: sg.Text
    screenshot_status_label_element: sg.Text


class BackgroundWorker:
    thread = None
    queue = collections.deque()
    exit = False

    def __init__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.exit:
            if self.queue:
                (action, args) = self.queue.popleft()
                try:
                    action(*args)
                except Exception as e:
                    logging.error(e)

            else:
                time.sleep(.05)

    def enqueue(self, action, *args):
        self.queue.append((action, args))


def get_image_bytes(img):
    bio = io.BytesIO()
    img.save(bio, 'PNG')
    return bio.getvalue()


def get_image_size(image_element: sg.Image):
    w = image_element.Widget
    return w.image.width(), w.image.height()


def get_pointer_position_on_image(image_element: sg.Image):
    w = image_element.Widget
    img_pos_x, img_pos_y = image_element.Position
    mouse_x, mouse_y = w.winfo_pointerxy()
    widget_width, widget_height = w.winfo_width(), w.winfo_height()
    image_width, image_height = w.image.width(), w.image.height()
    extra_width, extra_height = widget_width - image_width, widget_height - image_height
    padx, pady = extra_width - img_pos_x, extra_height - img_pos_y # not sure that's the right way but it works with current layout
    x, y = mouse_x - w.winfo_rootx() - padx, mouse_y - w.winfo_rooty() - pady
    return (x, y) if 0 <= x < image_width and 0 <= y < image_height else None


def get_coordinates_on_image(position_on_image, image_size, device_size):
    x, y = position_on_image
    x = x * device_size[0] // image_size[0]
    y = y * device_size[1] // image_size[1]
    return x, y


def capture_screenshot_action(model: Model, worker: BackgroundWorker, connector: AdbConnector):
    def action():
        try:
            model.screenshot_status_label_element.update("In progress...")
            model.main_image = connector.get_screenshot(raw=SCREENSHOT_RAW, pull=SCREENSHOT_PULL)
            img = model.main_image.copy()
            img.thumbnail(DISPLAY_MAX_SIZE)
            model.main_image_element.update(data=get_image_bytes(img))
            model.screenshot_status_label_element.update("")
        except:
            model.screenshot_status_label_element.update("Error")
    worker.enqueue(action)


def display_coordinates(model: Model):
    model.coord_label_element.update(value="")
    pos = get_pointer_position_on_image(model.main_image_element)
    if pos:
        x, y = get_coordinates_on_image(pos, get_image_size(model.main_image_element), model.device_screen_size)
        model.coord_label_element.update(value=f"x={x}, y={y}")
    else:
        pos = get_pointer_position_on_image(model.zoom_image_element)
        if pos:
            x, y = get_coordinates_on_image(pos, get_image_size(model.zoom_image_element), (model.zoom_radius * 2, model.zoom_radius * 2))
            x, y = x + model.zoom_center[0] - model.zoom_radius, y + model.zoom_center[1] - model.zoom_radius
            model.coord_label_element.update(value=f"x={x}, y={y}")


def update_zoom_image(model: Model):
    x, y = model.zoom_center
    zr = model.zoom_radius
    zoom = model.main_image.crop((x - zr, y - zr, x + zr, y + zr))
    zoom = zoom.resize((500, 500), resample=Image.NEAREST)
    model.zoom_image_element.update(data=get_image_bytes(zoom))


def layout_col_main_image(model: Model):
    model.main_image = Image.new('RGB', model.device_screen_size, START_COLOR)
    img = model.main_image.copy()
    img.thumbnail(DISPLAY_MAX_SIZE)
    model.main_image_element = sg.Image(data=get_image_bytes(img),
                                        enable_events=True,
                                        key=KEY_BUTTON_IMG_CLICK)
    screenshot_btn = sg.B("Get Screenshot", key=KEY_BUTTON_SCREENSHOT, size=(30, 1))
    model.screenshot_status_label_element = sg.T("", size=(20, 1))
    col = sg.Column(layout=[[screenshot_btn, model.screenshot_status_label_element], [model.main_image_element], [model.coord_label_element]],
                    element_justification='center')
    return col


def layout_col_zoom_image(model: Model):
    model.zoom_image = Image.new('RGB', (500, 500), START_COLOR)
    model.zoom_image_element = sg.Image(data=get_image_bytes(model.zoom_image))
    close_button = sg.B("Close", key=KEY_BUTTON_ZOOM_FRM_CLOSE)
    zoom_slider = sg.Slider(default_value=50, range=(ZOOM_RADIUS_MIN, ZOOM_RADIUS_MAX), disable_number_display=True,
                            orientation='h', resolution=10, enable_events=True, key=KEY_SLIDER_ZOOM)
    col = sg.Column(layout=[[zoom_slider, close_button],
                            [model.zoom_image_element]],
                           visible=False,
                           element_justification='center')
    model.zoom_column = col
    return col


def main():
    model = Model()
    worker = BackgroundWorker()

    logging.info(f"Connecting to device {config.phone_ip}")
    connector = AdbConnector(ip=config.phone_ip, adbkey_path=config.adbkey_path, output_dir=config.output_dir)
    width, height = connector.screen_width(), connector.screen_height()
    logging.info(f"Device screen size is {width} x {height}")
    model.device_screen_size = (width, height)
    model.zoom_center = (0, 0)
    model.zoom_radius = 50

    model.coord_label_element = sg.Text(size=(30, 1))
    layout = [
        [layout_col_main_image(model), layout_col_zoom_image(model)]
    ]
    window = sg.Window("Tapiocas' Sandbox", layout)

    while True:
        event, values = window.read(timeout=100)
        if event == sg.TIMEOUT_KEY:
            display_coordinates(model)
        elif event in (None, 'Exit'):
            break
        elif event == KEY_BUTTON_IMG_CLICK:
            pos = get_pointer_position_on_image(model.main_image_element)
            if pos:
                model.zoom_center = get_coordinates_on_image(pos, get_image_size(model.main_image_element), model.device_screen_size)
                update_zoom_image(model)
                model.zoom_column.update(visible=True)
        elif event == KEY_BUTTON_SCREENSHOT:
            capture_screenshot_action(model, worker, connector)
        elif event == KEY_BUTTON_ZOOM_FRM_CLOSE:
            model.zoom_column.update(visible=False)
        elif event == KEY_SLIDER_ZOOM:
            model.zoom_radius = values[KEY_SLIDER_ZOOM]
            update_zoom_image(model)

    window.close()


if __name__ == "__main__":
    config = config_manager.get_configuration()
    log_manager.initialize_log(config.log_dir, log_level=config.log_level)
    main()
