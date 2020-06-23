# solver with gui, for more options
import config_manager
import log_manager
from adb_connector import AdbConnector
import hungrycat as hc
import hungrycat_coords as hcc
import PySimpleGUI as sg
from enum import Enum, unique, auto
import logging


@unique
class Keys(Enum):
    BUTTON_SOLVE_FROM_SCREENSHOT = auto()
    RADIO_5_BY_5 = auto()
    RADIO_5_BY_10 = auto()
    RADIO_10_BY_10 = auto()
    RADIO_10_BY_15 = auto()
    SLIDER_BRUSHES = auto()


def layout_sliders():
    r_5_5 = sg.Radio("5 by 5", group_id="SIZE", key=Keys.RADIO_5_BY_5)
    r_5_10 = sg.Radio("5 by 10", group_id="SIZE", key=Keys.RADIO_5_BY_10)
    r_10_10 = sg.Radio("10 by 10", group_id="SIZE", key=Keys.RADIO_10_BY_10)
    r_10_15 = sg.Radio("10 by 15", group_id="SIZE", key=Keys.RADIO_10_BY_15, default=True)
    slider = sg.Slider(default_value=4, range=(2, 4), orientation='h', resolution=1, enable_events=True, key=Keys.SLIDER_BRUSHES)
    return sg.Column([[sg.T("Shape:"), r_5_5, r_5_10, r_10_10, r_10_15], [sg.T("Brushes:"), slider]])


def layout_buttons():
    parse_and_solve = sg.B("Screenshot and Solve", key=Keys.BUTTON_SOLVE_FROM_SCREENSHOT)
    return sg.Column([[parse_and_solve]])


def get_shape(key: Keys, nb_brush):
    if key == Keys.RADIO_10_BY_15 and nb_brush == 4:
        return hcc.Grid10By15b4()
    if key == Keys.RADIO_10_BY_10 and nb_brush == 4:
        return hcc.Grid10By10b4()
    if key == Keys.RADIO_10_BY_10 and nb_brush == 3:
        return hcc.Grid10By10b3()
    if key == Keys.RADIO_10_BY_10 and nb_brush == 2:
        return hcc.Grid10By10b2()
    logging.error("shape not implemented")
    return None


def run():
    layout = [
        [layout_sliders()],
        [layout_buttons()]
    ]
    window = sg.Window("HC", layout)
    window.read(timeout=1)

    while True:
        event, values = window.read(timeout=100)
        if event == sg.TIMEOUT_KEY:
            pass
        elif event in (None, 'Exit'):
            break
        elif event == Keys.BUTTON_SOLVE_FROM_SCREENSHOT:
            image_level = hc.get_level_image(save_image=False, load_image=False)
            shape_key = None
            for key in [Keys.RADIO_10_BY_15, Keys.RADIO_10_BY_10, Keys.RADIO_5_BY_10, Keys.RADIO_5_BY_5]:
                if values[key]:
                    shape_key = key
                    break
            shape = get_shape(shape_key, values[Keys.SLIDER_BRUSHES])
            ch, rh = hc.parse_headers(image_level, shape)
            solver.solve_level(shape, ch, rh)
            # solver.push_solution_cell_by_cell()
            solver.push_solution_line_by_line()


if __name__ == "__main__":
    hc.config = config_manager.get_configuration(config_folder='../../config')
    log_manager.initialize_log(hc.config.log_dir, log_level=hc.config.log_level)
    hc.connector = AdbConnector(ip=hc.config.phone_ip, adbkey_path=hc.config.adbkey_path, output_dir=hc.config.output_dir)
    hc.load_samples()
    solver = hc.Solver(hc.connector)

    run()

