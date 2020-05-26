import os
import io
import time
import re
import tempfile
from PIL import Image
import cv2
import numpy as np
import argparse
import logging
from datetime import datetime


from adb_shell.adb_device import AdbDeviceTcp
from adb_shell.auth.sign_pythonrsa import PythonRSASigner

from tapiocas.constants import *
from tapiocas import log_manager
from tapiocas import config_manager

logger = logging.getLogger(__name__)
logging.getLogger('adb_shell').setLevel(logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)


class AdbConnectorEventManager:
    """
    event manager methods are in case you need to use multi-touch or precise swipes
         (tap or press would not work, and swipe has inertia)
    press, release and move methods are enqueuing a command.
    flush() has to be called to send the whole thing
    todo: device-specific so that code needs a rework. Used to work with ZTE Axon7
        currently needs command-line "adb shell -- getevent -lt" to study device behaviour
    """
    def __init__(self):
        self.slots = 0
        self.curr_slot = 0
        self.commands = []

    def event_press(self, slot, x, y):
        if slot != self.curr_slot:
            self.commands.append(f'3 47 {slot}')
        self.commands.append(f'3 57 {slot}')
        if self.slots == 0:
            self.commands.append(f'1 330 1')
            self.commands.append(f'1 325 1')
        self.commands.append(f'3 53 {x}')
        self.commands.append(f'3 54 {y}')
        self.commands.append(f'3 58 50')
        self.commands.append(f'3 50 5')
        self.commands.append(f'0 0 0')
        self.slots += 1
        self.curr_slot = slot

    def event_release(self, slot):
        self.slots -= 1
        if slot != self.curr_slot:
            self.commands.append(f'3 47 {slot}')
        self.commands.append('3 57 -1')
        if self.slots == 0:
            self.commands.append('1 330 0')
            self.commands.append('1 325 0')
        self.commands.append('0 0 0')
        self.curr_slot = slot

    def event_move(self, slot, x, y):
        if slot != self.curr_slot:
            self.commands.append(f'3 47 {slot}')
        self.commands.append(f'3 53 {x}')
        self.commands.append(f'3 54 {y}')
        self.commands.append(f'0 0 0')
        self.curr_slot = slot

    def event_flush(self, device_name):
        command = ';'.join([f'sendevent {device_name} {c}' for c in self.commands])
        self.commands.clear()
        return command


class AdbConnector:
    """
    wraps adb_shell functions for simpler automation
    """
    def __init__(self, ip=None, need_auth=True, device_name="", auto_reconnect_seconds=60, adbkey_path=None, output_dir=None):
        self.ip = ip
        self.device_name = device_name
        self.need_auth = need_auth
        self.auto_reconnect_seconds = auto_reconnect_seconds
        self.adbkey_path = adbkey_path
        self.output_dir = output_dir

        self._event_handler = AdbConnectorEventManager()
        self._last_connected_time = time.time()
        self._device_resolution = None
        self._device = None

    def _connect(self):
        """
        in my experience, it was better to connect, send commands, disconnect right away
        rather than connect once and send many commands for hours => getting very slow at some point
        Now added auto-reconnect feature with argument auto_reconnect_seconds defaulting to 1 min
        """
        now = time.time()
        if self._device and self._last_connected_time + self.auto_reconnect_seconds < now:
            self._disconnect()

        if self._device:
            return

        logger.debug(f"connecting to {self.ip}")
        self._last_connected_time = now
        self._device = AdbDeviceTcp(self.ip, default_timeout_s=self.auto_reconnect_seconds)
        if not self.need_auth:
            self._device.connect(auth_timeout_s=0.1)
        else:
            with open(os.path.expanduser(self.adbkey_path)) as f:
                private_key = f.read()
            signer = PythonRSASigner('', private_key)
            self._device.connect(rsa_keys=[signer], auth_timeout_s=0.1)

        logger.debug(f"connected")

    def _disconnect(self):
        self._device.close()
        self._device = None
        logger.debug("disconnected")

    def _shell(self, command, **kwargs):
        logger.debug(f"shell {command}")
        return self._device.shell(command, **kwargs)

    def _pull(self, from_file, to_file):
        logger.debug(f"pull {from_file} to {to_file}")
        return self._device.pull(from_file, to_file)

    def tap(self, x, y, wait_ms=100):
        """
        tap the screen and force wait 100ms by default to simulate real taps if several in a row
        :param x:
        :param y:
        :param wait_ms:
        :return:
        """
        self._connect()
        self._shell(f'{CMD_SHELL_TAP} {x:.0f} {y:.0f}')
        self.wait(wait_ms)

    def press(self, x, y, time_ms=0, wait_ms=0):
        """
        long tap, implemented with a static swipe
        :param x:
        :param y:
        :param time_ms:
        :param wait_ms:
        :return:
        """
        self._connect()
        self._shell(f'{CMD_SHELL_SWIPE} {x:.0f} {y:.0f} {x:.0f} {y:.0f} {time_ms}')
        self.wait(wait_ms)

    def swipe(self, x1, y1, x2, y2, time_ms=0, wait_ms=0):
        """
        swipe from point (x1, y1) to point (x2, y2) in the span of time_ms
        careful, swipe has inertia, so if it is used to scroll a screen for example,
        screen will most likely keep scrolling at the end of the swipe
        To avoid that effect, use the event_move() instead
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param time_ms:
        :param wait_ms:
        :return:
        """
        self._connect()
        self._shell(f'{CMD_SHELL_SWIPE} {x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f} {time_ms}')
        self.wait(wait_ms)

    @staticmethod
    def wait(time_ms):
        """
        to wait some time. Not relying on adb, just convenient
        :param time_ms:
        :return:
        """
        if time_ms > 0:
            time.sleep(time_ms / 1000.0)

    @DeprecationWarning
    def event_press(self, slot, x, y):
        self._event_handler.event_press(slot, x, y)

    @DeprecationWarning
    def event_move(self, slot, x, y):
        self._event_handler.event_move(slot, x, y)

    @DeprecationWarning
    def event_release(self, slot):
        self._event_handler.event_release(slot)

    @DeprecationWarning
    def event_flush(self):
        self._connect()
        self._device.shell(self._event_handler.event_flush(self.device_name))

    # too much info: debug only
    def print_all_process_info(self):
        self._connect()
        processes_with_focus = self._shell(CMD_WINDOWS_DUMP)
        logger.debug(processes_with_focus)

    # todo: does not work anymore with latest android versions?
    @DeprecationWarning
    def process_has_focus(self, process_name):
        self._connect()
        all_processes = self._shell(f"{CMD_WINDOWS_DUMP} | grep -i {process_name} | grep -i mcurrentfocus")
        return len(all_processes) > 0

    def listen(self):
        self._connect()
        for line in self._device.streaming_shell(CMD_GET_EVENT):
            yield line

    # no way currently to pass an exit-predicate to adb-shell to exit the stream
    # so we just close the connection
    def abort_listening(self):
        self._disconnect()

    def _get_screen_resolution(self):
        if not self._device_resolution:
            self._connect()
            header_width_x_height = self._shell(CMD_WM_SIZE)
            self._device_resolution = tuple(map(int, re.findall("\d+", header_width_x_height)))
        return self._device_resolution

    def screen_width(self):
        return self._get_screen_resolution()[0]

    def screen_height(self):
        return self._get_screen_resolution()[1]

    def get_screenshot_pil(self, raw=False, pull=True) -> Image:
        if pull:
            if raw:
                return self._get_screenshot_raw_pull_file_pil()
            return self._get_screenshot_png_pull_file_pil()
        if raw:
            return self._get_screenshot_raw_stream_pil()
        return self._get_screenshot_png_stream_pil()

    def get_screenshot_opencv(self, raw=False, pull=False) -> np.ndarray:
        if pull:
            if raw:
                return self._get_screenshot_raw_pull_file_opencv()
            return self._get_screenshot_png_pull_file_opencv()
        if raw:
            return self._get_screenshot_raw_stream_opencv()
        return self._get_screenshot_png_stream_opencv()

    @staticmethod
    def get_temp_remote_filepath(extension):
        random_part = next(tempfile._get_candidate_names())
        return os.path.join(REMOTE_SCREENSHOT_DIRECTORY, f"screenshot_adb_{random_part}.{extension}")

    def get_temp_local_filepath(self, extension):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        return os.path.realpath(f"{self.output_dir}/screenshot.{extension}")

    def _raw_to_pil(self, raw) -> Image:
        return Image.frombuffer('RGBA', (self.screen_width(), self.screen_height()), raw[12:], 'raw', 'RGBX', 0, 1)

    def _raw_to_opencv(self, raw) -> np.ndarray:
        array = np.frombuffer(raw[16:], np.uint8)
        rgb_image = array.reshape((self.screen_height(), self.screen_width(), 4))
        # open_cv works with BGRA images
        bgr_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR)
        return bgr_image

    def _get_screenshot_png_pull_file_pil(self) -> Image:
        png_local_filepath = self._get_screenshot_png_pull_file()
        return Image.open(png_local_filepath)

    def _get_screenshot_png_pull_file_opencv(self) -> np.ndarray:
        png_local_filepath = self._get_screenshot_png_pull_file()
        return cv2.imread(png_local_filepath)

    def _get_screenshot_raw_pull_file_pil(self):
        raw_local_filepath = self._get_screenshot_raw_pull_file()
        with open(raw_local_filepath, 'rb') as f:
            raw = f.read()
        return self._raw_to_pil(raw)

    def _get_screenshot_raw_pull_file_opencv(self):
        raw_local_filepath = self._get_screenshot_raw_pull_file()
        with open(raw_local_filepath, 'rb') as f:
            raw = f.read()
        return self._raw_to_opencv(raw)

    def _get_screenshot_png_stream_pil(self) -> Image:
        stream = self._get_screenshot_png_stream()
        return Image.open(io.BytesIO(stream))

    def _get_screenshot_png_stream_opencv(self) -> np.ndarray:
        stream = self._get_screenshot_png_stream()
        array = np.frombuffer(stream, np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def _get_screenshot_raw_stream_pil(self) -> Image:
        stream = self._get_screenshot_raw_stream()
        return self._raw_to_pil(stream)

    def _get_screenshot_raw_stream_opencv(self) -> np.ndarray:
        stream = self._get_screenshot_raw_stream()
        return self._raw_to_opencv(stream)

    def _get_screenshot_png_pull_file(self) -> str:
        self._connect()
        png_remote_filepath = self.get_temp_remote_filepath("png")
        png_local_filepath = self.get_temp_local_filepath("png")
        start = time.perf_counter()
        self._shell(f"{CMD_SCREENSHOT_PNG} {png_remote_filepath}")
        end = time.perf_counter()
        logging.debug(f"png image screenshot took {end-start:.2f} seconds")
        self._pull(png_remote_filepath, png_local_filepath)
        start, end = end, time.perf_counter()
        logging.debug(f"png image pulled in {end-start:.2f} seconds")
        self._shell(f"rm {png_remote_filepath}")
        start, end = end, time.perf_counter()
        logging.debug(f"remote image deleted in {end-start:.2f} seconds")
        return png_local_filepath

    def _get_screenshot_raw_pull_file(self) -> str:
        self._connect()
        raw_remote_filepath = self.get_temp_remote_filepath("raw")
        raw_local_filepath = self.get_temp_local_filepath("raw")
        start = time.perf_counter()
        self._shell(f"{CMD_SCREENSHOT_RAW} {raw_remote_filepath}")
        end = time.perf_counter()
        logging.debug(f"raw image screenshot took {end-start:.2f} seconds")
        self._pull(raw_remote_filepath, raw_local_filepath)
        start, end = end, time.perf_counter()
        logging.debug(f"raw image pulled in {end-start:.2f} seconds")
        self._shell(f"rm {raw_remote_filepath}")
        start, end = end, time.perf_counter()
        logging.debug(f"remote image deleted in {end-start:.2f} seconds")
        return raw_local_filepath

    # todo: use exec-out instead of shell, because sometimes shell + no-decoding seems to be missing some bytes
    def _get_screenshot_png_stream(self):
        self._connect()
        raw = self._shell(CMD_SCREENSHOT_PNG, decode=False)
        return raw

    # todo: use exec-out instead of shell, because sometimes shell + no-decoding seems to be missing some bytes
    def _get_screenshot_raw_stream(self):
        self._connect()
        raw = self._shell(CMD_SCREENSHOT_RAW, decode=False)
        return raw


def run(config: config_manager.Configuration):
    log_manager.initialize_log(config.log_dir, log_file_name="adb_connector", log_level=config.log_level)

    logging.info('Starting adb connector')
    logging.info('Pid is {0}'.format(os.getpid()))
    logging.info('Log folder {0}'.format(config.log_dir))
    logging.info('Today is {0}'.format(str(datetime.today())))
    logging.info(f'Connecting to {config.phone_ip}')

    connector = AdbConnector(ip=config.phone_ip, adbkey_path=config.adbkey_path, output_dir=config.output_dir)
    width = connector.screen_width()
    height = connector.screen_height()
    logging.info(f"Screen is {width}x{height}")

    """
    on complex images, raw screenshot will be way faster than png screenshot
    but on wifi connection (through home router), pulling the data is very slow, 
    and raw image pull takes way longer than png image pull (because more data)
    Therefore png still seems to be the better choice for wifi, 
    but raw could be faster once adb-shell has usb connection implemented
    """
    for raw in (False, True):
        for pull in (False, True):
            start = time.perf_counter()
            image = connector.get_screenshot_pil(raw=raw, pull=pull)
            end = time.perf_counter()
            logging.info(f"image [raw={raw} pull={pull}] captured in {end-start:.2f} seconds")
            image.show()
    for raw in (False, True):
        for pull in (False, True):
            start = time.perf_counter()
            image = connector.get_screenshot_opencv(raw=raw, pull=pull)
            end = time.perf_counter()
            logging.info(f"image [raw={raw} pull={pull}] captured in {end-start:.2f} seconds")
            cv2.imshow(f"image [raw={raw} pull={pull}]", image)
            cv2.waitKey(5_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Android bot')
    parser.add_argument("--phone_ip", "-i", type=str, help='Ip of your phone')
    parser.add_argument("--config_file", "-c", type=str, help='Config file', default=CUSTOM_CONFIG_FILE)
    parser.add_argument("--log_level", "-l", help='Config file')
    argument = parser.parse_args()

    configuration = config_manager.get_configuration(argument.config_file)

    if argument.phone_ip:
        configuration.phone_ip = argument.phone_ip
    if argument.log_level:
        configuration.log_level = argument.log_level

    run(configuration)
