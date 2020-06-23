import os
import random
import time
import numpy as np
import cv2
import logging
from multiprocessing import Process

import config_manager
import log_manager
from adb_connector import AdbConnector
import text_reco
import image_filters

threshold_filter = image_filters.BinaryThresholdFilter()
invert_filter = image_filters.InvertFilter()


ORANGE_BUTTON_XY = (290, 1600)
BLUE_BUTTON_XY = (780, 1600)
ORANGE_BUTTON_BGR = (17, 201, 253)
BLUE_BUTTON_BGR = (254, 167, 24)

BATTLE_LIFE_RECT = (95, 1280, 1039, 1280)
HOME_LIFE_RECT = (95, 167, 1039, 167)
LIFE_COLOR_BGR = [95, 0, 192]

BATTLE_MONEY_RECT = (90, 1330, 400, 1388)
HOME_MONEY_RECT = (89, 216, 400, 274)

EXIT_ADVERT_XY = (240, 2095)
CHECK_ADVERT_RECT = (515, 2082, 567, 2106)

LAST_TAXI_XY = (1000, 1910)
TAXI_AVAILABLE_COLOR_BGR = (53, 0, 164)
START_BATTLE_XY = (520, 1880)

HOME_CHEAPEST_FOOD_XY = (200, 1650)
HOME_EXIT_XY = (550, 1900)

CAR_XY = (640, 555)
CAR_BGR = (194, 119, 170)
CAR_ENTER_XY = (530, 850)
CAR_EXIT_XY = (550, 1900)
CAR_TAP_RECT = (104, 1417, 990, 1704)

PARALLEL_PUNCH_KICK = 5
PARALLEL_PUNCH_KICK_WAIT_MS = 100

PARALLEL_CAR_TAP = 8
PARALLEL_CAR_TAP_WAIT_MS = 25


class MultiTap:
    def __init__(self, configuration: config_manager.Configuration):
        self.config = configuration
        self._processes = None
        self._connectors = None

    def _create_connector(self):
        c = AdbConnector(ip=self.config.phone_ip, adbkey_path=self.config.adbkey_path, output_dir=self.config.output_dir)

        return c

    def start_tapping(self, tap_coord_delegate, count, wait_ms):
        self.stop_tapping()
        for i in range(count):
            c = self._create_connector()
            c._connect()
            self._connectors.append(c)

        for c in self._connectors:
            def action():
                try:
                    while True:
                        xy = tap_coord_delegate()
                        c.tap(*xy, wait_ms=wait_ms)
                except Exception as e:
                    logging.exception(e)
            p = Process(target=action)
            self._processes.append(p)
            p.start()
            time.sleep(1 / count)

    def stop_tapping(self):
        if self._connectors:
            for c in self._connectors:
                c._disconnect()
        if self._processes:
            for p in self._processes:
                p.kill()
        self._processes = []
        self._connectors = []


def punch_or_kick_xy():
    return ORANGE_BUTTON_XY if random.randint(0, 1) else BLUE_BUTTON_XY


def car_rect_xy():
    x = random.randint(CAR_TAP_RECT[0], CAR_TAP_RECT[2])
    y = random.randint(CAR_TAP_RECT[1], CAR_TAP_RECT[3])
    return x, y


# if the home button oval is shown, then it's an advert
# otherwise it's all black
def is_showing_advert(image):
    x1, y1, x2, y2 = CHECK_ADVERT_RECT
    advert = np.sum(image[y1:y2+1, x1:x2+1]) > 1
    if advert:
        logging.info("Advert detected")
    return advert


def exit_advert(connector: AdbConnector):
    connector.tap(*EXIT_ADVERT_XY, wait_ms=500)


def eat_food(connector: AdbConnector):
    image = connector.get_screenshot_opencv()
    while get_life_pct(image, HOME_LIFE_RECT) < 99:
        connector.tap(*HOME_CHEAPEST_FOOD_XY)
        image = connector.get_screenshot_opencv()


def exit_home(connector: AdbConnector):
    time.sleep(.5)
    connector.tap(*HOME_EXIT_XY, wait_ms=500)


def can_go_to_battle(image):
    return (image[LAST_TAXI_XY[1]][LAST_TAXI_XY[0]] == TAXI_AVAILABLE_COLOR_BGR).all()


def is_in_battle(image):
    orange_button_displayed = (image[ORANGE_BUTTON_XY[1]][ORANGE_BUTTON_XY[0]] == ORANGE_BUTTON_BGR).all()
    blue_button_displayed = (image[BLUE_BUTTON_XY[1]][BLUE_BUTTON_XY[0]] == BLUE_BUTTON_BGR).all()
    return orange_button_displayed and blue_button_displayed


def go_to_battle(connector: AdbConnector):
    connector.tap(*START_BATTLE_XY, wait_ms=1500)


def can_dismantle_car(image):
    return (image[CAR_XY[1]][CAR_XY[0]] == CAR_BGR).all()


def enter_garage(connector: AdbConnector):
    connector.tap(*CAR_ENTER_XY, wait_ms=1000)


def dismantle_car(multi_tap: MultiTap):
    multi_tap.start_tapping(car_rect_xy, count=PARALLEL_CAR_TAP, wait_ms=PARALLEL_CAR_TAP_WAIT_MS)
    time.sleep(13)
    multi_tap.stop_tapping()
    time.sleep(3)


def exit_garage(connector: AdbConnector):
    connector.tap(*CAR_EXIT_XY, wait_ms=1000)


def check_garage_macro(connector: AdbConnector, multi_tap: MultiTap):
    image = connector.get_screenshot_opencv()
    if can_dismantle_car(image):
        enter_garage(connector)
        dismantle_car(multi_tap)
        exit_garage(connector)


def get_life_pct(image, life_rect):
    x1 = life_rect[0]
    x2 = life_rect[2]
    y = life_rect[1]
    life_bar = image[y][x1:x2 + 1]
    s = np.sum(life_bar == LIFE_COLOR_BGR) / 3
    p = int(s * 100 / (x2 - x1 + 1))
    return p


def get_money_amount(image, money_rect):
    x1, y1, x2, y2 = money_rect
    money_bar = image[y1:y2 + 1, x1:x2 + 1]
    money_bar = threshold_filter.apply(money_bar)
    money_bar = invert_filter.apply(money_bar)
    t = text_reco.find_text(money_bar, psm=text_reco.PSM.SINGLE_BLOCK, whitelist='0123456789')
    fp = os.path.join(config.output_dir, "pkpp", f"money_crop_{money_rect[1]}_{time.time()}__{t}.png")
    print(fp)
    cv2.imwrite(fp, money_bar)
    return t


def run(connector: AdbConnector):
    i = 0
    start_money = None
    start_run_money = None
    multi_tap = MultiTap(config)
    check_garage_macro(connector, multi_tap)
    while True:
        i += 1

        # go to battle if possible
        image = connector.get_screenshot_opencv()
        if not is_in_battle(image):
            if not can_go_to_battle(image):
                time.sleep(5)
                continue
            go_to_battle(connector)
            life_rect, money_rect = HOME_LIFE_RECT, HOME_MONEY_RECT
        else:
            life_rect, money_rect = BATTLE_LIFE_RECT, BATTLE_MONEY_RECT

        life = get_life_pct(image, life_rect)
        money = get_money_amount(image, money_rect)
        if money.isnumeric():
            money = int(money)
            if start_run_money is not None:
                print(f"MONEY EARNED SINCE LAST RUN: {money - start_run_money}")
            if start_money is not None:
                print(f"MONEY EARNED SINCE THE BEGINNING: {money - start_money}")
            else:
                start_money = money
            start_run_money = money
        print(f"Run #{i}:")
        print(f"LIFE = {life}%     MONEY = {money}")

        # fight until advert
        multi_tap.start_tapping(punch_or_kick_xy, count=PARALLEL_PUNCH_KICK, wait_ms=PARALLEL_PUNCH_KICK_WAIT_MS)
        while not is_showing_advert(image):
            time.sleep(10 if life > 50 else 5 if life > 10 else 2)
            image = connector.get_screenshot_opencv()
            life = get_life_pct(image, BATTLE_LIFE_RECT)
            print(f"LIFE = {life}%")
        multi_tap.stop_tapping()
        time.sleep(1)

        while is_showing_advert(image):
            exit_advert(connector)
            image = connector.get_screenshot_opencv()

        eat_food(connector)
        exit_home(connector)
        check_garage_macro(connector, multi_tap)


if __name__ == "__main__":
    config = config_manager.get_configuration(config_folder='../../config')
    log_manager.initialize_log(config.log_dir, log_level=config.log_level)
    adb_connector = AdbConnector(ip=config.phone_ip, adbkey_path=config.adbkey_path, output_dir=config.output_dir)
    run(adb_connector)
