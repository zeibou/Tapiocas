import io
import PySimpleGUI as sg
from PIL import Image
import logging
import collections
import threading
import time
import random
from enum import Enum, unique, auto

from adb_connector import AdbConnector
import config_manager
import log_manager


@unique
class Keys(Enum):
    BUTTON_RECORD = auto(),
    MULTILINE_RECORD = auto(),
    BUTTON_SCREENSHOT = auto(),
    IMAGE_MAIN = auto(),
    RADIO_GROUP_IMAGE_MAIN_CLICK = auto(),
    RADIO_BTN_IMAGE_MAIN_ZOOM = auto(),
    RADIO_BTN_IMAGE_MAIN_TAP = auto(),
    IMAGE_ZOOM = auto(),
    SLIDER_ZOOM = auto(),
    BUTTON_ZOOM_CLOSE = auto(),
    COLUMN_ZOOM = auto(),
    LABEL_STATUS = auto(),
    LABEL_COORD = auto(),


DISPLAY_MAX_SIZE = (300, 600)
SCREENSHOT_RAW = False
SCREENSHOT_PULL = False

ZOOM_RADIUS_MIN = 10
ZOOM_RADIUS_MAX = 100
START_COLOR = (random.randint(25, 100), random.randint(25, 100), random.randint(25, 100))
RECORDING_COLOR = "tan1"
NO_RECORDING_COLOR = "lightsteelblue2"


class Model:
    device_screen_size: (int, int)
    zoom_center: (int, int)
    zoom_radius: int

    recording: bool
    recording_stopped: bool

    main_image: Image
    zoom_image = Image

    window: sg.Window


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
            model.window[Keys.LABEL_STATUS].update("Screenshot in progress...")
            model.main_image = connector.get_screenshot(raw=SCREENSHOT_RAW, pull=SCREENSHOT_PULL)
            img = model.main_image.copy()
            img.thumbnail(DISPLAY_MAX_SIZE)
            model.window[Keys.IMAGE_MAIN].update(data=get_image_bytes(img))
            update_zoom_image(model)
            model.window[Keys.LABEL_STATUS].update("")
        except Exception as e:
            model.window[Keys.LABEL_STATUS].update("Error")
            logging.error(e)
    worker.enqueue(action)


def send_tap_action(coords, model: Model, worker: BackgroundWorker, connector: AdbConnector):
    def action():
        try:
            model.window[Keys.LABEL_STATUS].update(f"Sending tap to {coords}")
            connector.tap(*coords, wait_ms=0)
            model.window[Keys.LABEL_STATUS].update("")
        except Exception as e:
            model.window[Keys.LABEL_STATUS].update("Error")
            logging.error(e)
    worker.enqueue(action)


def record_events_action(model: Model, worker: BackgroundWorker, connector: AdbConnector):
    def action():
        try:
            model.recording = True
            model.recording_stopped = False
            model.window[Keys.BUTTON_RECORD].update(text="Stop recording")
            model.window[Keys.MULTILINE_RECORD].update(background_color=RECORDING_COLOR)
            model.window[Keys.LABEL_STATUS].update(f"Recording events...")
            for t in connector.listen():
                model.window[Keys.MULTILINE_RECORD].print(t)
            model.window[Keys.LABEL_STATUS].update("")
        except Exception as e:
            # exception is expected when we normally abort the recording, because we close the connection
            if model.recording_stopped:
                model.window[Keys.LABEL_STATUS].update("")
            else:
                model.window[Keys.LABEL_STATUS].update("Error")
                logging.error(e)
        finally:
            model.recording = False
            model.window[Keys.BUTTON_RECORD].update(text="Record events")
            model.window[Keys.MULTILINE_RECORD].update(background_color=NO_RECORDING_COLOR)
    worker.enqueue(action)


def get_pointer_pos_in_device_coordinates(model: Model):
    pos = get_pointer_position_on_image(model.window[Keys.IMAGE_MAIN])
    if pos:
        return get_coordinates_on_image(pos, get_image_size(model.window[Keys.IMAGE_MAIN]), model.device_screen_size)
    else:
        pos = get_pointer_position_on_image(model.window[Keys.IMAGE_ZOOM])
        if pos:
            zoom_diameter = int(model.zoom_radius * 2)
            x, y = get_coordinates_on_image(pos, get_image_size(model.window[Keys.IMAGE_ZOOM]), (zoom_diameter, zoom_diameter))
            return x + model.zoom_center[0] - model.zoom_radius, y + model.zoom_center[1] - model.zoom_radius


def display_pointer_pos_in_device_coordinates(model: Model):
    pos = get_pointer_pos_in_device_coordinates(model)
    if pos:
        x, y = pos
        model.window[Keys.LABEL_COORD].update(value=f"x={int(x)}  y={int(y)}")
    else:
        model.window[Keys.LABEL_COORD].update(value="")


def update_zoom_image(model: Model):
    x, y = model.zoom_center
    zr = model.zoom_radius
    zoom = model.main_image.crop((x - zr, y - zr, x + zr, y + zr))
    zoom = zoom.resize((500, 500), resample=Image.NEAREST)
    model.window[Keys.IMAGE_ZOOM].update(data=get_image_bytes(zoom))


def layout_col_action_panel():
    b = sg.B("Record events", key=Keys.BUTTON_RECORD, size=(40, 1))
    t = sg.Multiline(size=(75, 30), key=Keys.MULTILINE_RECORD, autoscroll=True, background_color=NO_RECORDING_COLOR)
    col = sg.Column(layout=[[b], [t]])
    return col


def layout_col_main_image_menu():
    screenshot_btn = sg.B("Get Screenshot", key=Keys.BUTTON_SCREENSHOT, size=(30, 1))
    status_label_element = sg.T("", size=(40, 1), key=Keys.LABEL_STATUS)
    r1 = sg.Radio("Zoom on click", key=Keys.RADIO_BTN_IMAGE_MAIN_ZOOM, group_id=Keys.RADIO_GROUP_IMAGE_MAIN_CLICK, default=True)
    r2 = sg.Radio("Tap on click", key=Keys.RADIO_BTN_IMAGE_MAIN_TAP, group_id=Keys.RADIO_GROUP_IMAGE_MAIN_CLICK)
    col = sg.Column(layout=[[status_label_element], [r1, r2], [screenshot_btn]])
    return col


def layout_col_main_image(model: Model):
    coord_label_element = sg.Text(size=(30, 1), justification='center', key=Keys.LABEL_COORD)
    model.main_image = Image.new('RGB', model.device_screen_size, START_COLOR)
    img = model.main_image.copy()
    img.thumbnail(DISPLAY_MAX_SIZE)
    main_image_element = sg.Image(data=get_image_bytes(img),
                                        enable_events=True,
                                        key=Keys.IMAGE_MAIN)
    col = sg.Column(layout=[[layout_col_main_image_menu()], [main_image_element], [coord_label_element]],
                    element_justification='center')
    return col


def layout_col_zoom_image(model: Model):
    model.zoom_image = Image.new('RGB', (500, 500), START_COLOR)
    zoom_image_element = sg.Image(data=get_image_bytes(model.zoom_image),
                                        enable_events=True,
                                        key=Keys.IMAGE_ZOOM)
    close_button = sg.B("Close", key=Keys.BUTTON_ZOOM_CLOSE)
    zoom_slider = sg.Slider(default_value=50, range=(ZOOM_RADIUS_MIN, ZOOM_RADIUS_MAX), disable_number_display=True,
                            orientation='h', resolution=10, enable_events=True, key=Keys.SLIDER_ZOOM)
    col = sg.Column(layout=[[zoom_slider, close_button],
                            [zoom_image_element]],
                           key=Keys.COLUMN_ZOOM,
                           visible=False,
                           element_justification='center')
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
    model.recording = False

    layout = [
        [layout_col_action_panel(), layout_col_main_image(model), layout_col_zoom_image(model)]
    ]
    window = sg.Window("Tapiocas' Sandbox", layout)
    model.window = window

    while True:
        event, values = window.read(timeout=100)
        if event == sg.TIMEOUT_KEY:
            display_pointer_pos_in_device_coordinates(model)
        elif event in (None, 'Exit'):
            break
        elif event in (Keys.IMAGE_MAIN, Keys.IMAGE_ZOOM):
            pos = get_pointer_pos_in_device_coordinates(model)
            if pos:
                if values[Keys.RADIO_BTN_IMAGE_MAIN_ZOOM]:
                    model.zoom_center = pos
                    update_zoom_image(model)
                    model.window[Keys.COLUMN_ZOOM].update(visible=True)
                elif values[Keys.RADIO_BTN_IMAGE_MAIN_TAP]:
                    send_tap_action(pos, model, worker, connector)
        elif event == Keys.BUTTON_SCREENSHOT:
            capture_screenshot_action(model, worker, connector)
        elif event == Keys.BUTTON_ZOOM_CLOSE:
            model.window[Keys.COLUMN_ZOOM].update(visible=False)
        elif event == Keys.SLIDER_ZOOM:
            model.zoom_radius = values[Keys.SLIDER_ZOOM]
            update_zoom_image(model)
        elif event == Keys.BUTTON_RECORD:
            if model.recording:
                # not sent to background worker because we want to abort the current action
                model.recording_stopped = True
                connector.abort_listening()
            else:
                record_events_action(model, worker, connector)

    window.close()


if __name__ == "__main__":
    config = config_manager.get_configuration()
    log_manager.initialize_log(config.log_dir, log_level=config.log_level)
    main()
