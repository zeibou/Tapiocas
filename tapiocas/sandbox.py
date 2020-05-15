import PySimpleGUI as sg
import cv2
import numpy as np
import logging
import collections
import threading
import time
from enum import Enum, unique, auto
import os

from adb_connector import AdbConnector
import config_manager
import log_manager
import image_filters as filters
import text_reco


@unique
class Keys(Enum):
    BUTTON_RECORD = auto()
    MULTILINE_RECORD = auto()
    BUTTON_SCREENSHOT = auto()
    BUTTON_SCREENSHOT_LIVE = auto()
    IMAGE_MAIN = auto()
    RADIO_GROUP_IMAGE_MAIN_CLICK = auto()
    RADIO_BTN_IMAGE_MAIN_ZOOM = auto()
    RADIO_BTN_IMAGE_MAIN_TAP = auto()
    IMAGE_ZOOM = auto()
    SLIDER_ZOOM = auto()
    BUTTON_ZOOM_CLOSE = auto()
    COLUMN_ZOOM = auto()
    LABEL_STATUS = auto()
    LABEL_COORD = auto()
    LIST_FILTERS_LIBRARY = auto()
    LIST_FILTERS_SELECTED = auto()
    BUTTON_FILTER_ADD = auto()
    BUTTON_FILTER_REMOVE = auto()
    INPUT_FILTER_VALUE = auto()
    BUTTON_FILTER_APPLY = auto()
    BUTTON_FILTER_DOWN = auto()
    BUTTON_FILTER_UP = auto()
    CHECKBOX_FILTER_ACTIVE = auto()
    CHECKBOX_ALL_FILTER_ACTIVE = auto()
    SPIN_CROP_TOP = auto()
    SPIN_CROP_LEFT = auto()
    SPIN_CROP_WIDTH = auto()
    SPIN_CROP_HEIGHT = auto()
    BUTTON_CROP_SAVE = auto()
    BUTTON_TESSERACT_CHECK = auto()
    INPUT_TESSERACT_WHITELIST = auto()
    COMBOBOX_TESSERACT_PSM = auto()
    MULTILINE_TESSERACT_RESULT = auto()


DISPLAY_SIZE_RATIO = 4
SCREENSHOT_RAW = False
SCREENSHOT_PULL = False

ZOOM_IMAGE_SIZE = 500
ZOOM_LEVELS = [-1, 250, 112, 50, 22, 10]
RECORDING_COLOR = "tan1"
NO_RECORDING_COLOR = "lightsteelblue2"

IMAGE_FILTERS = {f.name(): f for f in [
    filters.GrayFilter,
    filters.BlurFilter,
    filters.CannyFilter,
    filters.ContourFilter,
    filters.InvertFilter,
    filters.ThresholdFilter,
]}


class Model:
    window: sg.Window

    def __init__(self, device_screen_size):
        self.device_screen_size = device_screen_size
        self._zoom_center = (0, 0)
        self._zoom_width = 100
        self._zoom_height = 100
        self.zoom_mode = False
        self.recording = False
        self.recording_stopped = False
        self.live_screenshot = False
        start_image = np.random.randint(50, 150, (device_screen_size[1], device_screen_size[0], 3), np.uint8)
        self.screenshot_raw = start_image
        self.screenshot_filtered = start_image
        self.zoom_filtered = None
        self._screenshot_lock = threading.Lock()
        self._screenshot_raw_new = None
        self.filters = []
        self.apply_filters = True
        self._zoom_rectangle = None
        self._compute_zoom_rectangle()

    # called from background thread
    def enqueue_new_screenshot(self, img):
        self._screenshot_lock.acquire()
        self._screenshot_raw_new = img
        self._screenshot_lock.release()

    # called from main thread
    def check_for_new_screenshot(self):
        refresh = False
        self._screenshot_lock.acquire()
        if self._screenshot_raw_new is not None:
            self.screenshot_raw, self._screenshot_raw_new = self._screenshot_raw_new, None
            # if we detect a change of orientation, we reverse device_size and zoom shapes
            if self.device_screen_size[1] != self.screenshot_raw.shape[0]:
                self.orientation_changed()
            refresh = True
        self._screenshot_lock.release()
        return refresh

    def orientation_changed(self):
        self.device_screen_size = self.screenshot_raw.shape[:2][::-1]
        self._zoom_center = self._zoom_center[::-1]
        self._zoom_width, self._zoom_height = self._zoom_height, self._zoom_width
        self.window[Keys.SPIN_CROP_LEFT].update(values=[i for i in range(self.device_screen_size[0])])
        self.window[Keys.SPIN_CROP_TOP].update(values=[i for i in range(self.device_screen_size[1])])
        self.window[Keys.SPIN_CROP_WIDTH].update(values=[i for i in range(1, self.device_screen_size[0])])
        self.window[Keys.SPIN_CROP_HEIGHT].update(values=[i for i in range(1,  self.device_screen_size[1])])
        self._compute_zoom_rectangle(True)

    def update_zoom_center(self, center):
        self._zoom_center = center
        self._compute_zoom_rectangle(True)

    def update_zoom_radius(self, radius):
        # we override the current width / height
        if radius > 0:
            self._zoom_width = radius * 2
            self._zoom_height = radius * 2
        else:
            self._zoom_width = self.device_screen_size[0]
            self._zoom_height = self.device_screen_size[1]
        self._compute_zoom_rectangle(True)

    def update_crop(self, left, top, width, height):
        left = max(0, min(int(left), self.device_screen_size[0]-1))
        top = max(0, min(int(top), self.device_screen_size[1]-1))
        width = max(1, min(int(width), self.device_screen_size[0]-1))
        height = max(1, min(int(height), self.device_screen_size[1]-1))
        self._zoom_center = (left + (width // 2 + width % 2 - 1), top + (height // 2 + height % 2 - 1))
        self._zoom_width = width
        self._zoom_height = height
        self._compute_zoom_rectangle(True)

    def _compute_zoom_rectangle(self, update_spin_elements=False):
        zx, zy = self._zoom_center
        w, h = self._zoom_width, self._zoom_height
        # if w or h is even, the center pixel is on the upper left side (+1 pixel on the right / bottom side)
        # wl / wr is the number of pixel to the left / right of center, same for ht / hb
        wl, wr = w // 2 + w % 2 - 1, w // 2
        ht, hb = h // 2 + h % 2 - 1, h // 2
        # adjust center to stay inside image
        zx = min(max(zx, wl), self.device_screen_size[0] - wr - 1)
        zy = min(max(zy, ht), self.device_screen_size[1] - hb - 1)
        self._zoom_rectangle = (zx - wl, zy - ht), (zx + wr, zy + hb)
        if update_spin_elements:
            self.window[Keys.SPIN_CROP_LEFT].update(value=zx - wl)
            self.window[Keys.SPIN_CROP_TOP].update(value=zy - ht)
            self.window[Keys.SPIN_CROP_WIDTH].update(value=self._zoom_width)
            self.window[Keys.SPIN_CROP_HEIGHT].update(value=self._zoom_height)

    @property
    def zoom_rectangle(self):
        return self._zoom_rectangle

    @property
    def zoom_width(self):
        return self._zoom_width

    @property
    def zoom_height(self):
        return self._zoom_height


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
                    logging.exception(e)

            else:
                time.sleep(.05)

    def enqueue(self, action, *args):
        self.queue.append((action, args))


def get_image_thumbnail(img):
    thumbnail = cv2.resize(img, (0, 0), fx=1/DISPLAY_SIZE_RATIO, fy=1/DISPLAY_SIZE_RATIO, interpolation=cv2.INTER_AREA)
    return thumbnail


def get_image_bytes(img):
    return cv2.imencode('.png', img)[1].tobytes()


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
    # not sure that's the right way but it works with current layout
    padx, pady = extra_width - img_pos_x, extra_height - img_pos_y
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
            new_screenshot = connector.get_screenshot_opencv(raw=SCREENSHOT_RAW, pull=SCREENSHOT_PULL)
            model.enqueue_new_screenshot(new_screenshot)
            model.window[Keys.LABEL_STATUS].update("")
            if model.live_screenshot:
                capture_screenshot_action(model, worker, connector)
        except Exception as e:
            model.window[Keys.LABEL_STATUS].update("Error")
            logging.exception(e)
    worker.enqueue(action)


def send_tap_action(coords, model: Model, worker: BackgroundWorker, connector: AdbConnector):
    def action():
        try:
            model.window[Keys.LABEL_STATUS].update(f"Sending tap to {coords}")
            connector.tap(*coords, wait_ms=0)
            model.window[Keys.LABEL_STATUS].update("")
        except Exception as e:
            model.window[Keys.LABEL_STATUS].update("Error")
            logging.exception(e)
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
                logging.exception(e)
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
            ul, br = model.zoom_rectangle
            w, h = model.zoom_width, model.zoom_height
            x, y = get_coordinates_on_image(pos, get_image_size(model.window[Keys.IMAGE_ZOOM]), (w, h))
            return x + ul[0], y + ul[1]


def display_pointer_pos_in_device_coordinates(model: Model):
    pos = get_pointer_pos_in_device_coordinates(model)
    if pos:
        x, y = pos
        model.window[Keys.LABEL_COORD].update(value=f"x={int(x)}  y={int(y)}")
    else:
        model.window[Keys.LABEL_COORD].update(value="")


def update_main_image(model: Model):
    model.screenshot_filtered = apply_filters(model, model.screenshot_raw)
    img = get_image_thumbnail(model.screenshot_filtered)

    if model.zoom_mode:
        ul, br = model.zoom_rectangle
        h, w = img.shape[:2]
        ul = get_coordinates_on_image(ul, model.device_screen_size, (w, h))
        br = get_coordinates_on_image(br, model.device_screen_size, (w, h))
        cv2.rectangle(img, ul, br, (50, 50, 240, 240))
    model.window[Keys.IMAGE_MAIN].update(data=get_image_bytes(img))


def update_zoom_image(model: Model):
    ul, br = model.zoom_rectangle
    model.zoom_filtered = model.screenshot_filtered[ul[1]:br[1]+1, ul[0]:br[0]+1]
    w, h = model.zoom_width, model.zoom_height
    r = ZOOM_IMAGE_SIZE / max(h, w)
    h2, w2 = int(h * r), int(w * r)
    zoom = cv2.resize(model.zoom_filtered, (w2, h2), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(zoom, (0, 0), (w2 - 1, h2 - 1), (50, 50, 240, 240))
    model.window[Keys.IMAGE_ZOOM].update(data=get_image_bytes(zoom))


def save_crop_image(model: Model):
    x, y = model.zoom_rectangle[0]
    filename = f"crop_x{x}_y{y}_w{model.zoom_width}_h{model.zoom_height}.png"
    filepath = os.path.join(config.output_dir, filename)
    while True:
        filepath = sg.popup_get_text(title="Save as...", message="", size=(100, 1), default_text=filepath)
        if not filepath:
            return
        if not os.path.exists(filepath) or "Yes" == sg.popup_yes_no("File exists. Overwrite ?", title="Warning"):
            logging.info(f"Saving crop image to {filepath}")
            cv2.imwrite(filepath, model.zoom_filtered)
            return


def apply_filters(model: Model, image):
    if model.apply_filters:
        for f in model.filters:
            if f.enabled:
                try:
                    image = f.apply(image)
                except Exception as e:
                    logging.exception(e)
    return image


def layout_controls_tab_container(model: Model):
    filters_tab = sg.Tab('Image Filters', [[layout_col_filters_panel(model)]])
    recording_tab = sg.Tab('Events Recording', [[layout_col_action_panel()]])
    tab_group_layout = [[filters_tab, recording_tab]]
    return sg.Column(layout=[[sg.TabGroup(layout=tab_group_layout)]])


def layout_col_filters_panel(model: Model):
    available_filters = sg.Listbox(list(IMAGE_FILTERS.keys()), bind_return_key=True, size=(40, 8), key=Keys.LIST_FILTERS_LIBRARY)
    applied_filters = sg.Listbox(model.filters, enable_events=True, size=(40, 5), key=Keys.LIST_FILTERS_SELECTED)
    add_button = sg.Button("Add", key=Keys.BUTTON_FILTER_ADD, size=(25, 1))
    apply_button = sg.Button("Apply", key=Keys.BUTTON_FILTER_APPLY, size=(10, 1), disabled=True)
    remove_button = sg.Button("Remove", key=Keys.BUTTON_FILTER_REMOVE, size=(25, 1), disabled=True)
    value_input = sg.Input(key=Keys.INPUT_FILTER_VALUE, size=(12, 1), disabled=True)
    down_button = sg.Button("Down", key=Keys.BUTTON_FILTER_DOWN, size=(10, 1), disabled=True)
    up_button = sg.Button("Up", key=Keys.BUTTON_FILTER_UP, size=(10, 1), disabled=True)
    enabled_checkbox = sg.Checkbox("Enabled", key=Keys.CHECKBOX_FILTER_ACTIVE, enable_events=True, size=(25, 1), disabled=True)
    return sg.Column([[available_filters, add_button],
                      [applied_filters, sg.Column(layout=[[value_input, apply_button], [enabled_checkbox], [up_button, down_button], [remove_button]])]])


def handle_image_filters_options_visibility(model: Model):
    selected_indices = model.window[Keys.LIST_FILTERS_SELECTED].TKListbox.curselection()
    if selected_indices is None or len(selected_indices) != 1:
        show_value = show_apply = False
        show_remove = show_checkbox = False
        show_up = show_down = False
    else:
        selected_filter = model.filters[selected_indices[0]]
        show_remove = show_checkbox = True
        show_value = show_apply = selected_filter.value is not None
        show_up = len(model.filters) > 1 and selected_filter != model.filters[0]
        show_down = len(model.filters) > 1 and selected_filter != model.filters[-1]

    model.window[Keys.INPUT_FILTER_VALUE].update(disabled=not show_value)
    model.window[Keys.BUTTON_FILTER_APPLY].update(disabled=not show_apply)
    model.window[Keys.BUTTON_FILTER_REMOVE].update(disabled=not show_remove)
    model.window[Keys.BUTTON_FILTER_DOWN].update(disabled=not show_down)
    model.window[Keys.BUTTON_FILTER_UP].update(disabled=not show_up)
    model.window[Keys.CHECKBOX_FILTER_ACTIVE].update(disabled=not show_checkbox)


def on_filters_changed(model: Model, refresh_buttons, refresh_listbox):
    if refresh_buttons:
        handle_image_filters_options_visibility(model)
    if refresh_listbox:
        index = model.window[Keys.LIST_FILTERS_SELECTED].TKListbox.curselection()
        model.window[Keys.LIST_FILTERS_SELECTED].update(values=model.filters)
        model.window[Keys.LIST_FILTERS_SELECTED].TKListbox.selection_set(index)

    update_main_image(model)
    update_zoom_image(model)


def layout_col_action_panel():
    b = sg.B("Record events", key=Keys.BUTTON_RECORD, size=(40, 1))
    t = sg.Multiline(size=(75, 30), key=Keys.MULTILINE_RECORD, autoscroll=True, background_color=NO_RECORDING_COLOR)
    col = sg.Column(layout=[[b], [t]])
    return col


def layout_col_main_image_menu():
    screenshot_btn = sg.B("Get Screenshot", key=Keys.BUTTON_SCREENSHOT, size=(30, 1))
    live_btn = sg.Checkbox("Live", key=Keys.BUTTON_SCREENSHOT_LIVE, enable_events=True, size=(10, 1))
    status_label_element = sg.T("", size=(40, 1), key=Keys.LABEL_STATUS)
    r1 = sg.Radio("Zoom on click", key=Keys.RADIO_BTN_IMAGE_MAIN_ZOOM, group_id=Keys.RADIO_GROUP_IMAGE_MAIN_CLICK, default=True)
    r2 = sg.Radio("Tap on click", key=Keys.RADIO_BTN_IMAGE_MAIN_TAP, group_id=Keys.RADIO_GROUP_IMAGE_MAIN_CLICK)
    cb_filters = sg.Checkbox("Filters", default=True, key=Keys.CHECKBOX_ALL_FILTER_ACTIVE, enable_events=True)
    col = sg.Column(layout=[[status_label_element], [r1, r2, cb_filters], [screenshot_btn, live_btn]])
    return col


def layout_col_main_image(model: Model):
    coord_label_element = sg.Text(size=(30, 1), justification='center', key=Keys.LABEL_COORD)
    main_image_element = sg.Image(data=get_image_bytes(get_image_thumbnail(model.screenshot_raw)),
                                  enable_events=True,
                                  key=Keys.IMAGE_MAIN)
    col = sg.Column(layout=[[layout_col_main_image_menu()], [main_image_element], [coord_label_element]],
                    element_justification='center')
    return col


def layout_tab_zoom():
    zoom_slider = sg.Slider(default_value=3, range=(0, len(ZOOM_LEVELS) - 1), disable_number_display=True,
                            orientation='h', resolution=1, enable_events=True, size=(60, 20), key=Keys.SLIDER_ZOOM)
    label_line = [sg.T(f"  x{250 // z if z > 0 else 0}", size=(11, 1)) for z in ZOOM_LEVELS]
    slider_line = [zoom_slider]
    return sg.Tab('Zoom', [[sg.T()], label_line, slider_line])


def layout_tab_crop(model: Model):
    left_values = [i for i in range(model.device_screen_size[0])]
    width_values = [i for i in range(1, model.device_screen_size[0])]
    top_values = [i for i in range(model.device_screen_size[1])]
    height_values = [i for i in range(1, model.device_screen_size[1])]
    # should be updated when orientation changes
    crop_line_1 = [sg.T("Left:", size=(8, 1)),
                   sg.Spin(left_values, key=Keys.SPIN_CROP_LEFT, size=(8, 1), enable_events=True),
                   sg.T("Width:", size=(8, 1)),
                   sg.Spin(width_values, key=Keys.SPIN_CROP_WIDTH, size=(8, 1), enable_events=True)]
    crop_line_2 = [sg.T("Top:", size=(8, 1)),
                   sg.Spin(top_values, key=Keys.SPIN_CROP_TOP, size=(8, 1), enable_events=True),
                   sg.T("Height:", size=(8, 1)),
                   sg.Spin(height_values, key=Keys.SPIN_CROP_HEIGHT, size=(8, 1), enable_events=True)]

    return sg.Tab('Crop', [[sg.T()], crop_line_1, crop_line_2])


def layout_tab_text_reco():
    psm_combobox = sg.InputCombo(values=[v for v in text_reco.PSM],
                                 default_value=text_reco.PSM.SPARSE_TEXT,
                                 key=Keys.COMBOBOX_TESSERACT_PSM)
    label_whitelist = sg.T("    Whitelist:", tooltip="example: '0123456789' for numeric only, empty means no restriction")
    input_whitelist = sg.Input("", key=Keys.INPUT_TESSERACT_WHITELIST, size=(20, 1))
    ocr_button = sg.B("Find Text", key=Keys.BUTTON_TESSERACT_CHECK)
    result = sg.Multiline(size=(40, 3), key=Keys.MULTILINE_TESSERACT_RESULT, background_color="lightgray")
    inputs_col = sg.Column(layout=[[psm_combobox], [label_whitelist, input_whitelist], [ocr_button]])
    return sg.Tab('Text', [[inputs_col, result]])


def layout_col_zoom_image(model):
    zoom_image = np.zeros((ZOOM_IMAGE_SIZE, ZOOM_IMAGE_SIZE, 3), np.uint8)
    zoom_image_element = sg.Image(data=get_image_bytes(zoom_image),
                                  enable_events=True,
                                  key=Keys.IMAGE_ZOOM)
    close_button = sg.B("Close", key=Keys.BUTTON_ZOOM_CLOSE)
    save_button = sg.B(f"Save as", key=Keys.BUTTON_CROP_SAVE)

    tab_group_layout = [[layout_tab_zoom(), layout_tab_crop(model), layout_tab_text_reco()]]
    col = sg.Column(layout=[[sg.TabGroup(layout=tab_group_layout)],
                            [zoom_image_element],
                            [save_button, close_button]],
                    key=Keys.COLUMN_ZOOM,
                    element_justification='center')
    return col


def main():
    worker = BackgroundWorker()

    logging.info(f"Connecting to device {config.phone_ip}")
    connector = AdbConnector(ip=config.phone_ip, adbkey_path=config.adbkey_path, output_dir=config.output_dir)
    width, height = connector.screen_width(), connector.screen_height()
    logging.info(f"Device screen size is {width} x {height}")
    model = Model((width, height))

    layout = [
        [layout_controls_tab_container(model), layout_col_main_image(model), layout_col_zoom_image(model)]
    ]
    window = sg.Window("Tapiocas' Sandbox", layout)
    model.window = window
    # read once to initialize elements
    window.read(timeout=1)

    # update zoom to initialize crop spin elements
    model.zoom_mode = True
    model.update_zoom_center((width // 2, height // 2))
    update_main_image(model)
    update_zoom_image(model)

    while True:
        event, values = window.read(timeout=100)
        new_screenshot = model.check_for_new_screenshot()
        if new_screenshot:
            update_main_image(model)
            if model.zoom_mode:
                update_zoom_image(model)

        if event == sg.TIMEOUT_KEY:
            display_pointer_pos_in_device_coordinates(model)
        elif event in (None, 'Exit'):
            break
        elif event in (Keys.IMAGE_MAIN, Keys.IMAGE_ZOOM):
            pos = get_pointer_pos_in_device_coordinates(model)
            if pos:
                if values[Keys.RADIO_BTN_IMAGE_MAIN_ZOOM]:
                    model.zoom_mode = True
                    model.update_zoom_center(pos)
                    update_main_image(model)
                    update_zoom_image(model)
                    model.window[Keys.COLUMN_ZOOM].update(visible=True)
                elif values[Keys.RADIO_BTN_IMAGE_MAIN_TAP]:
                    send_tap_action(pos, model, worker, connector)
        elif event == Keys.BUTTON_SCREENSHOT:
            capture_screenshot_action(model, worker, connector)
        elif event == Keys.BUTTON_SCREENSHOT_LIVE:
            model.live_screenshot = values[Keys.BUTTON_SCREENSHOT_LIVE]
        elif event == Keys.BUTTON_CROP_SAVE:
            save_crop_image(model)
        elif event == Keys.BUTTON_ZOOM_CLOSE:
            model.zoom_mode = False
            model.window[Keys.COLUMN_ZOOM].update(visible=False)
            update_main_image(model)
        elif event == Keys.SLIDER_ZOOM:
            model.update_zoom_radius(ZOOM_LEVELS[int(values[Keys.SLIDER_ZOOM])])
            update_main_image(model)
            update_zoom_image(model)
        elif event in (Keys.SPIN_CROP_LEFT, Keys.SPIN_CROP_TOP, Keys.SPIN_CROP_WIDTH, Keys.SPIN_CROP_HEIGHT):
            model.update_crop(values[Keys.SPIN_CROP_LEFT],
                              values[Keys.SPIN_CROP_TOP],
                              values[Keys.SPIN_CROP_WIDTH],
                              values[Keys.SPIN_CROP_HEIGHT])
            update_main_image(model)
            update_zoom_image(model)
        elif event == Keys.BUTTON_TESSERACT_CHECK:
            psm = values[Keys.COMBOBOX_TESSERACT_PSM]
            whitelist = values[Keys.INPUT_TESSERACT_WHITELIST]
            txt = text_reco.find_text(model.zoom_filtered, psm=psm, whitelist=whitelist)
            window[Keys.MULTILINE_TESSERACT_RESULT].update(value=txt)
        elif event == Keys.BUTTON_RECORD:
            if model.recording:
                # not sent to background worker because we want to abort the current action
                model.recording_stopped = True
                connector.abort_listening()
            else:
                record_events_action(model, worker, connector)
        elif event == Keys.BUTTON_FILTER_ADD or event == Keys.LIST_FILTERS_LIBRARY:
            selected_filter = values[Keys.LIST_FILTERS_LIBRARY]
            if selected_filter is not None and len(selected_filter) == 1:
                if selected_filter[0] in IMAGE_FILTERS:
                    model.filters.append(IMAGE_FILTERS[selected_filter[0]]())
                    window[Keys.LIST_FILTERS_SELECTED].update(values=model.filters)
                    on_filters_changed(model, True, False)
        elif event == Keys.BUTTON_FILTER_REMOVE:
            selected = values[Keys.LIST_FILTERS_SELECTED]
            if selected is not None:
                for s in selected:
                    model.filters.remove(s)
                window[Keys.LIST_FILTERS_SELECTED].update(values=model.filters)
                on_filters_changed(model, True, False)
        elif event == Keys.LIST_FILTERS_SELECTED:
            handle_image_filters_options_visibility(model)
            selected = values[Keys.LIST_FILTERS_SELECTED]
            if selected is not None and len(selected) == 1:
                value = selected[0].value
                window[Keys.INPUT_FILTER_VALUE].update(value=value if value is not None else '')
                window[Keys.CHECKBOX_FILTER_ACTIVE].update(value=selected[0].enabled)
        elif event == Keys.BUTTON_FILTER_APPLY:
            selected = values[Keys.LIST_FILTERS_SELECTED]
            if selected is not None and len(selected) == 1:
                selected[0].set_value(values[Keys.INPUT_FILTER_VALUE])
                on_filters_changed(model, False, True)
        elif event == Keys.BUTTON_FILTER_UP:
            index = window[Keys.LIST_FILTERS_SELECTED].TKListbox.curselection()
            if len(index) == 1 and index[0] > 0:
                i = index[0]
                model.filters[i-1], model.filters[i] = model.filters[i], model.filters[i-1]
                window[Keys.LIST_FILTERS_SELECTED].update(values=model.filters)
                window[Keys.LIST_FILTERS_SELECTED].TKListbox.selection_set((i-1,))
                on_filters_changed(model, True, False)
        elif event == Keys.BUTTON_FILTER_DOWN:
            index = window[Keys.LIST_FILTERS_SELECTED].TKListbox.curselection()
            if len(index) == 1 and index[0] < len(model.filters) - 1:
                i = index[0]
                model.filters[i+1], model.filters[i] = model.filters[i], model.filters[i+1]
                window[Keys.LIST_FILTERS_SELECTED].update(values=model.filters)
                window[Keys.LIST_FILTERS_SELECTED].TKListbox.selection_set((i+1,))
                on_filters_changed(model, True, False)
        elif event == Keys.CHECKBOX_FILTER_ACTIVE:
            selected = values[Keys.LIST_FILTERS_SELECTED]
            if selected is not None and len(selected) == 1:
                selected[0].enabled = values[Keys.CHECKBOX_FILTER_ACTIVE]
                on_filters_changed(model, True, True)
        elif event == Keys.CHECKBOX_ALL_FILTER_ACTIVE:
            model.apply_filters = values[Keys.CHECKBOX_ALL_FILTER_ACTIVE]
            on_filters_changed(model, False, False)

    window.close()


if __name__ == "__main__":
    config = config_manager.get_configuration()
    log_manager.initialize_log(config.log_dir, log_level=config.log_level)
    main()
