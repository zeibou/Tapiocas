import os
import io
import time
import re
import tempfile
from PIL import Image
import argparse

from adb_shell.adb_device import AdbDeviceTcp
from adb_shell.auth.sign_pythonrsa import PythonRSASigner

from constants import *
import logging
import log_manager
import json
from datetime import datetime

logger = logging.getLogger(__name__)
logging.getLogger('adb_shell').setLevel(logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)

# useful terminal commands:
# connect to phone via usb:
##   adb start-server

# list connected devices
##   adb devices

# listen to events
##   adb shell -- getevent -lt;

# Then press the phone's screen and get data like:
# [ 1059536.835861] /dev/input/event2: EV_KEY       BTN_TOUCH            DOWN
# [ 1059536.835861] /dev/input/event2: EV_ABS       ABS_MT_TRACKING_ID   0000eb79            
# [ 1059536.835861] /dev/input/event2: EV_ABS       ABS_MT_POSITION_X    00000213            
# [ 1059536.835861] /dev/input/event2: EV_ABS       ABS_MT_POSITION_Y    00000420 
# "/dev/input/event2" will be needed to send events

# kill the server before using this module. Can't connect otherwise
##   adb kill-server


# KitKat+ devices require authentication
# see this link in case it's not automatic with adb_shell
# https://stackoverflow.com/questions/33005354/trouble-installing-m2crypto-with-pip-on-os-x-macos

# connection over wifi
# https://futurestud.io/tutorials/how-to-debug-your-android-app-over-wifi-without-root



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
    def __init__(self, ip=None, need_auth=True, device_name="", auto_reconnect_seconds=60):
        self.ip = ip
        self.device_name = device_name
        self.need_auth = need_auth
        self.auto_reconnect_seconds = auto_reconnect_seconds

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
            with open(os.path.expanduser('~/.android/adbkey')) as f:
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


    def tap(self, x, y, wait_ms = 100):
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

    def press(self, x, y, time_ms = 0, wait_ms = 0):
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

    def swipe(self, x1, y1, x2, y2, time_ms = 0, wait_ms = 0):
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
        logger.debug(x1, y1, x2, y2)
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

    # todo: needs PR merged in adb_shell module
    def listen(self):
        self._connect()
        for line in self._device.streaming_shell(CMD_GET_EVENT):
            logger.debug(line)

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

    def get_screenshot(self, raw=False, pull=True) -> Image:
        if pull:
            if raw:
                return self._get_screenshot_raw_pull_file()
            return self._get_screenshot_png_pull_file()
        if raw:
            raise Exception("Not implemented")
        return self._get_screenshot_png_stream()

    @staticmethod
    def get_temp_remote_filepath(extension):
        random_part = next(tempfile._get_candidate_names())
        return os.path.join(REMOTE_SCREENSHOT_DIRECTORY, f"screenshot_adb_{random_part}.{extension}")

    @staticmethod
    def get_temp_local_filepath(extension):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        return os.path.realpath(f"{OUTPUT_DIR}/screenshot.{extension}")

    def _get_screenshot_png_pull_file(self) -> Image:
        self._connect()
        png_remote_filepath = self.get_temp_remote_filepath("png")
        png_local_filepath = self.get_temp_local_filepath("png")
        self._shell(f"{CMD_SCREENSHOT_PNG} {png_remote_filepath}")
        self._pull(png_remote_filepath, png_local_filepath)
        self._shell(f"rm {png_remote_filepath}")
        return Image.open(png_local_filepath)

    def _get_screenshot_raw_pull_file(self) -> Image:
        self._connect()
        raw_remote_filepath = self.get_temp_remote_filepath("raw")
        raw_local_filepath = self.get_temp_local_filepath("raw")
        self._shell(f"{CMD_SCREENSHOT_RAW} {raw_remote_filepath}")
        self._pull(raw_remote_filepath, raw_local_filepath)
        self._shell(f"rm {raw_remote_filepath}")
        with open(raw_local_filepath, 'rb') as f:
            raw = f.read()
        return Image.frombuffer('RGBA', (self.screen_height(), self.screen_width()), raw[12:], 'raw', 'RGBX', 0, 1)

    # todo: use exec-out instead of shell
    def _get_screenshot_png_stream(self) -> Image:
        self._connect()
        raw = self._shell(CMD_SCREENSHOT_PNG, decode=False)
        image = Image.open(io.BytesIO(raw))
        return image


def get_pixel(image, x, y):
    logger.debug(image.getpixel((x, y)))

def timeit(method, n, *args):
    logger.info(f"Running {method}")
    t = time.time()
    for _ in range(n):
        method(*args)

    logger.info(time.time() - t)


def run(phone_ip, config_file, log_level):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"No file found at [{config_file}]")
    with open(config_file, "r") as fin:
        config = json.load(fin)
    log_folder = config["log_dir"]
    log_manager.initialize_log(log_folder)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.info('Starting adb connector')
    logger.info('Pid is {0}'.format(os.getpid()))
    logger.info('Log folder {0}'.format(log_folder))
    logger.info('Today is {0}'.format(str(datetime.today())))



    connector = AdbConnector(ip=phone_ip)
    width = connector.screen_width()
    height = connector.screen_height()
    logger.debug(f"Screen is {width}x{height}")


    connector.tap(1000, 2000)
    # connector.listen()

    # connector.print_all_process_info()

    # timeit(connector.get_screenshot, 2, True, True)
    # timeit(connector.get_screenshot, 2, False, True)
    timeit(connector.get_screenshot, 2, False, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Android bot')
    parser.add_argument("--phone_ip", "-i", type=str, required=True, help='Ip of your phone')
    parser.add_argument("--config_file", "-c", type=str, help='Config file',
                        default="./config/adbc.json")
    parser.add_argument("--log_level", "-l", help='Config file',
                        default="INFO", type=lambda x: LOG_DICO[x],choices=LOG_DICO.keys())
    argument = parser.parse_args()
    run(argument.phone_ip, argument.config_file, argument.log_level)

    # print(image.getpixel((100, 100)))

    # connector.listen(timeout_ms=10_000)

    # connector.event_press(0, 500, 500)
    # connector.event_flush()
    # connector.wait(3000)
    # connector.event_release(0)
    # connector.event_flush()
