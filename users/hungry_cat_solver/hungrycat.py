import os
import time
import logging
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import config_manager
import log_manager
from adb_connector import AdbConnector
from image_filters import CannyFilter, BlurFilter, InvertFilter


canny_filter = CannyFilter()
blur_filter = BlurFilter()
blur_filter.set_value(5)
invert_filter = InvertFilter()

MIN_SSIM_SCORE = 0.4
IMVERT_COLORS = False

"""
Works with device screen size: 1080 x 2160
"""

#  circle colors (alternate color on rows => 2 colors for each item)
CIRCLE_1 = (193, 193, 193)
CIRCLE_2 = (183, 183, 183)

# brush color position
BRUSH_1_XY = (290, 2030)
BRUSH_2_XY = (500, 2030)
BRUSH_3_XY = (710, 2030)
BRUSH_4_XY = (920, 2030)
BRUSHES = [BRUSH_1_XY, BRUSH_2_XY, BRUSH_3_XY, BRUSH_4_XY]

# header cells positions (we need precision for better image comparison, and cells are not linearly spaced)
COLUMN_HEADERS_X = [305, 381, 459, 535, 612, 682, 759, 836, 913, 989]
COLUMN_HEADERS_Y = [381, 444, 508, 571]

ROW_HEADERS_X = [48, 111, 175, 238]
ROW_HEADERS_Y = [638, 714, 791, 868, 944, 1015, 1092, 1168, 1246, 1322, 1392, 1469, 1546, 1623, 1700]

HEADER_W = 50
HEADER_H = 50

# solution cells approx positions:
GRID_LEFT = 290
GRID_TOP = 623
GRID_RIGHT = 1050
GRID_BOTTOM = 1760
GRID_SHAPE = (10, 15)  # number of cells per row, per column

# image recognition
SAMPLES_DIR = './hungry_cat_solver/digits_samples'
SAMPLES = [None] * 16

# milliseconds per cell, too fast may swipe too far
SWIPE_SPEED = 50


def load_samples():
    for f in os.listdir(SAMPLES_DIR):
        if f.endswith('png'):
            fp = os.path.join(SAMPLES_DIR, f)
            index = int(f.split('.')[0])
            im = cv2.imread(fp)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            SAMPLES[index] = im


# returns lots of 'True' if the circle color is detected.
# May be a false positive is the brush has exactly the same color
def get_circle_mask(image):
    return np.all(image == CIRCLE_1[::-1], axis=-1) | np.all(image == CIRCLE_2[::-1], axis=-1)


class SsimScore:
    def __init__(self, score, value):
        self.score = score
        self.value = value


class Header:
    def __init__(self, i, j, value, circled, brush_id):
        self.i = i
        self.j = j
        self.value = value
        self.circled = circled  # (contiguous points)
        self.brush_id = brush_id
        self.ssim_scores = []  # for other possible parsings

    def __repr__(self):
        return  f"{'(' if self.circled else ''}{self.value}{')' if self.circled else ''}"


class Cell:
    def __init__(self, i, j, v):
        self.i = i
        self.j = j
        self.v = v

    @property
    def screen_pos(self):
        cell_w = (GRID_RIGHT - GRID_LEFT) / GRID_SHAPE[0]
        cell_h = (GRID_BOTTOM - GRID_TOP) / GRID_SHAPE[1]
        x = int(GRID_LEFT + (self.i + .5) * cell_w)
        y = int(GRID_TOP + (self.j + .5) * cell_h)
        return x, y

    def __repr__(self):
        return  f"{self.i + 1} => {self.j + 1}"


class Step:
    def __init__(self, cell_1, cell_2, brush_id):
        self.cell_1 = cell_1
        self.cell_2 = cell_2
        self.brush_id = brush_id
        self.distance = abs(cell_1.i - cell_2.i) + abs(cell_1.j - cell_2.j)

    def __repr__(self):
        return  f"Brush #{self.brush_id + 1}: {self.cell_1} => {self.cell_2}"


class Solver:
    def __init__(self, connector: AdbConnector):
        self.connector = connector
        self.row_headers = None
        self.col_headers = None
        self.solution = None
        self.solution_by_step = None

    def solve_level(self, col_headers, row_headers):
        self.col_headers = col_headers
        self.row_headers = row_headers
        t = time.perf_counter()
        logging.info("Looking for solution")
        self.dfs_solution()
        t = time.perf_counter() - t
        logging.info(f"Found solution in {t} seconds")

    # let's solve by recursion
    # grid model: 150 cells, 4 possible colors, use 1 bit per possible color.
    # for each recursive path we can pass a single 600 bits number state
    # if down to 1 => color picked for that path.
    # if down to 0 => not possible, wrong path.
    # if 2 or more => try all possibilities
    def dfs_solution(self):
        width = GRID_SHAPE[0]
        height = GRID_SHAPE[1]
        col_headers = self.col_headers
        row_headers = self.row_headers

        def init_grid():
            g = 0
            for j in range(height):
                for i in range(width):
                    for b in range(4):
                        hc = col_headers[b][i]
                        hr = row_headers[b][j]
                        if hc.value and hr.value:
                            p = 4 * (j * width + i) + b
                            g |= 1 << p
            return g

        # returns a tuple of 4 zeroes / ones
        def get_cell(g, i, j):
            pos = 4 * (j * width + i)
            return [1 if g & (1 << pos + k) else 0 for k in range(4)]

        def set_color(g, i, j, brush_id):
            # update cell color
            pos = 4 * (j * width + i)
            g &= ~(0b1111 << pos)  # set all to zero first
            g |= 1 << pos + brush_id
            return g

        def remove_color(g, i, j, brush_id):
            pos = 4 * (j * width + i)
            g &= ~(1 << pos + brush_id)
            return g

        def max_simplify(g):
            g2 = simplify_grid(g)
            while g2 != g:
                g = g2
                g2 = simplify_grid(g)
            return g2

        # tries to simplify grid possibilities with a few rules
        # returns None if grid is invalid
        def simplify_grid(g):
            g = try_place_contiguous(g, False)
            g = try_place_contiguous(g, True)
            g = check_completed_and_update(g, False)
            g = check_completed_and_update(g, True)
            return g

        def try_place_contiguous(g, is_row):
            if g is None:
                return None
            _range = height if is_row else width
            _headers = row_headers if is_row else col_headers
            _row_or_col = "row" if is_row else "col"
            for index in range(_range):
                for header in sorted([_headers[b][index] for b in range(4)], key=lambda h: -h.value):
                    if header.value > 1 and header.circled:
                        b = header.brush_id
                        blocks = find_brush_blocks(g, b, index, is_row)
                        invalid_blocks = [block for block in blocks if block[1] - block[0] + 1 < header.value]
                        valid_blocks = [block for block in blocks if block[1] - block[0] + 1 >= header.value]
                        # if no valid block, then the current grid is impossible
                        # if only one block valid, we can validate the center part of it
                        # if more than one valid block possible, then we also check whether t
                        # he color is 'sure' in one of those blocks, in which case other blocks are invalid
                        if len(valid_blocks) == 0:
                            return None
                        if len(valid_blocks) == 1:
                            block = valid_blocks[0]
                            certified = certified_cells_in_block(g, b, index, block, is_row)
                            # we can just certify the middle of the block
                            block_len = block[1] - block[0] + 1
                            margin = block_len - header.value
                            start_block, end_block = block[0] + margin, block[1] - margin + 1
                            # some part of the block is certified. if it's a side, we know where it ends
                            if certified:
                                if block[0] in certified:
                                    start_block, end_block = block[0] + 1, block[0] + header.value
                                elif block[1] in certified:
                                    start_block, end_block = block[1] - header.value + 1, block[1]
                            for k in range(start_block, end_block):
                                if is_row:
                                    g = set_color(g, k, index, b)
                                else:
                                    g = set_color(g, index, k, b)

                        elif len(valid_blocks) > 1:
                            only_valid_block = None
                            for block in valid_blocks:
                                certified = certified_cells_in_block(g, b, index, block, is_row)
                                if certified:
                                    # then other blocks are invalid
                                    only_valid_block = block
                                    break
                            if only_valid_block:
                                invalid_blocks += [b for b in valid_blocks if b != only_valid_block]

                        for block in invalid_blocks:
                            # logging.debug(f"impossible block placement on {_row_or_col} {index}: brush #{b+1} , {block[0] + 1} -> {block[1] + 1}")
                            for k in range(block[0], block[1] + 1):
                                if is_row:
                                    g = remove_color(g, k, index, b)
                                else:
                                    g = remove_color(g, index, k, b)
            return g

        # returns a list of possible contiguous blocks for brush b
        def find_brush_blocks(g, b, index, is_row):
            _range = width if is_row else height
            _get_cell = (lambda x: get_cell(g, x, index)) if is_row else (lambda x: get_cell(g, index, x))
            start = -1
            blocks = []
            for k in range(_range):
                color = _get_cell(k)[b]
                if color and start == -1:
                    start = k
                elif not color and start >= 0:
                    blocks.append((start, k - 1))
                    start = -1
            if start >= 0:
                blocks.append((start, _range - 1))
            return blocks

        # check if a brush is for sure in that block
        def certified_cells_in_block(g, b, index, block, is_row):
            certified = []
            _get_cell = (lambda x: get_cell(g, x, index)) if is_row else (lambda x: get_cell(g, index, x))
            for k in range(block[0], block[1] + 1):
                color = _get_cell(k)
                if color[b] and sum(color) == 1:
                    certified.append(k)
            return certified

        # we check all lines and all columns
        # for each color x line x column: we count the number of sure cells
        # if it's equal to the header's value, we disable color in other cells
        def check_completed_and_update(g, is_row):
            _range = height if is_row else width
            _headers = row_headers if is_row else col_headers
            for index in range(_range):
                if g is None:
                    return None
                if is_row:
                    color_counts = [get_cell(g, k, index) for k in range(width)]
                else:
                    color_counts = [get_cell(g, index, k) for k in range(height)]
                for b in range(4):
                    if _headers[b][index].value > 0:
                        color_count = sum(c[b] for c in color_counts if sum(c) == 1)
                        if color_count == _headers[b][index].value:
                            if is_row:
                                for k in range(width):
                                    if color_counts[k][b] == 1 and sum(color_counts[k]) > 1:
                                        g = remove_color(g, k, index, b)
                            else:
                                for k in range(height):
                                    if color_counts[k][b] == 1 and sum(color_counts[k]) > 1:
                                        g = remove_color(g, index, k, b)
            return g

        # to get rid of wrong recursion solutions faster, at each new line we check it is valid
        # we check also every column that is fully solved
        def check_line(g, j, is_row):
            if is_row:
                color_counts = [get_cell(g, k, j) for k in range(width)]
            else:
                color_counts = [get_cell(g, j, k) for k in range(height)]
            for b, header_list in enumerate(row_headers if is_row else col_headers):
                header = header_list[j]
                if header.value != sum(k[b] for k in color_counts):
                    return False
                if header.value > 1:
                    blocks = find_brush_blocks(g, b, j, is_row)
                    if header.circled and len(blocks) != 1:
                        return False
                    if not header.circled and len(blocks) == 1:
                        return False
            return True

        def get_rows_recur_order():
            # we sort by headers with highest circled value / lowest possibilities
            # so that recursive solving is faster
            scores = []
            for i in range(height):
                headers = [row_headers[b][i] for b in range(4) if row_headers[b][i].value > 0]
                circled = [h.value for h in headers if h.circled]
                score = 100 * max(circled) if circled else 0 + 10 * (4 - len(headers))
                scores.append(score)

            return [i for i, k in sorted(enumerate(scores), key=lambda s: -s[1])]

        # g is the grid model, r and c are row and col models (contain currently picked colors per row / col)
        # very slow...
        def recur(g, i, row_index):
            if i == 0 and row_index > 0:
                if not check_line(g, recur_j[row_index - 1], True):
                    return None
                # we also try to simplify at every new recursive line, helps checking for invalid paths
                g = simplify_grid(g)
                if g is None:
                    return None
            if row_index >= height:
                for i in range(width):
                    if not check_line(g, i, False):
                        return None
                # we cleared the board, g is a solution
                return g
            j = recur_j[row_index]
            colors = get_cell(g, i, j)
            next_i, next_row_index = (i + 1, row_index) if i + 1 < width else (0, row_index + 1)
            if sum(colors) == 0:
                return None
            else:
                # we want to perform checks
                for b in range(4):
                    if colors[b]:
                        hg = set_color(g, i, j, b)
                        s = recur(hg, next_i, next_row_index)
                        if s is not None:
                            return s

        def grid_to_solution(g):
            if not g:
                raise ValueError("could not find solution")
            solution = []
            for j in range(height):
                for i in range(width):
                    brushes = get_cell(g, i, j)
                    b_ids = [i for i in range(4) if brushes[i]]
                    if len(b_ids) != 1:
                        pass
                        logging.warning(f"Cell [{i}, {j}] not solved. Will not push color on that cell.")
                    else:
                        solution.append(Cell(i, j, b_ids[0]))
            return solution

        def grid_to_solution_by_step(g):
            if not g:
                raise ValueError("could not find solution")

            def block_to_step(brush, i1, j1, i2, j2):
                return Step(Cell(i1, j1, brush), Cell(i2, j2, brush), brush)

            solution_steps = []
            for j in range(height):
                for b in range(4):
                    for block in find_brush_blocks(g, b, j, True):
                        solution_steps.append(block_to_step(b, block[0], j, block[1], j))
            for i in range(width):
                for b in range(4):
                    for block in find_brush_blocks(g, b, i, False):
                        solution_steps.append(block_to_step(b, i, block[0], i, block[1]))

            solution_steps.sort(key=lambda s: -s.distance)

            # we discard steps that don't add anything
            solution_steps_uniques = []
            grid = np.ones((15, 10), np.bool)
            for step in solution_steps:
                g = grid.copy()
                for j in range(step.cell_1.j, step.cell_2.j + 1):
                    for i in range(step.cell_1.i, step.cell_2.i + 1):
                        g[j][i] = 0
                if np.sum(g) != np.sum(grid):
                    grid = g
                    solution_steps_uniques.append(step)

            return solution_steps_uniques

        grid = init_grid()
        grid = max_simplify(grid)
        recur_j = get_rows_recur_order()
        grid = recur(grid, 0, 0)
        self.solution = grid_to_solution(grid)
        self.solution_by_step = grid_to_solution_by_step(grid)

    # slow but works on unfinished solutions as well (for debug or step by step)
    def push_solution_cell_by_cell(self):
        for brush_id in range(4):
            bx, by = BRUSHES[brush_id]
            self.connector.tap(bx, by)
            for c in [c for c in self.solution if c.v == brush_id]:
                self.connector.tap(*c.screen_pos)

    # fast but only works on finished solution
    def push_solution_line_by_line(self):
        for brush_id in range(4):
            bx, by = BRUSHES[brush_id]
            self.connector.tap(bx, by)
            for s in [s for s in self.solution_by_step if s.brush_id == brush_id]:
                self.connector.swipe(*s.cell_1.screen_pos, *s.cell_2.screen_pos, s.distance * SWIPE_SPEED)


def parse_column_headers(image):
    headers = []
    for j in range(4):
        h_line = []
        for i in range(GRID_SHAPE[0]):
            x = COLUMN_HEADERS_X[i]
            y = COLUMN_HEADERS_Y[j]
            crop = image[y:y + HEADER_H + 1, x:x + HEADER_W + 1]
            h_line.append(crop_to_header(crop, i, j, j, False))
        headers.append(h_line)
    return headers


def parse_row_headers(image):
    headers = []
    for i in range(4):
        v_line = []
        for j in range(GRID_SHAPE[1]):
            x = ROW_HEADERS_X[i]
            y = ROW_HEADERS_Y[j]
            crop = image[y:y + HEADER_H + 1, x:x + HEADER_W + 1]
            v_line.append(crop_to_header(crop, i, j, i, True))
        headers.append(v_line)
    return headers


def crop_filter(crop, circled):
    # sometimes inverting colors helps recognizing contours
    if IMVERT_COLORS:
        crop = invert_filter.apply(crop)
    crop = canny_filter.apply(crop)
    if circled:
        # remove corners that have the circle contrast
        remove_circle_corners(crop)
    # we blur the canny image a few times to have thicker borders
    crop = blur_filter.apply(blur_filter.apply(blur_filter.apply(crop)))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop


def remove_circle_corners(crop):
    k = 8
    mask_ur = np.tril(np.ones((k, k), dtype=np.uint8), -1)
    mask_bl = mask_ur.T
    mask_ul = mask_bl[::-1]
    mask_br = mask_ur[::-1]
    crop[:k, :k, 0] *= mask_ul
    crop[:k, :k, 1] *= mask_ul
    crop[:k, :k, 2] *= mask_ul
    crop[:k, -k:, 0] *= mask_ur
    crop[:k, -k:, 1] *= mask_ur
    crop[:k, -k:, 2] *= mask_ur
    crop[-k:, :k, 0] *= mask_bl
    crop[-k:, :k, 1] *= mask_bl
    crop[-k:, :k, 2] *= mask_bl
    crop[-k:, -k:, 0] *= mask_br
    crop[-k:, -k:, 1] *= mask_br
    crop[-k:, -k:, 2] *= mask_br


def crop_to_header(crop, i, j, brush_id, is_row):
    ci_mask = get_circle_mask(crop)
    circled = np.sum(ci_mask) > 100
    header = Header(i, j, -1, circled, brush_id)
    filtered = crop_filter(crop, circled)

    # uncomment to generate samples / debug
    filepath = os.path.join(config.output_dir, "hungry_cat_crops", f"crop_{'R' if is_row else 'C'}_x{i+1}_y{j+1}.png")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, filtered)

    # 0 is an easy case, and we want to enforce it with a high score
    max_intensity = np.max(filtered)
    if max_intensity == 0:
        header.ssim_scores.append(SsimScore(100, 0))

    for s_id, sample in enumerate(SAMPLES):
        if sample is not None:
            score, _ = ssim(filtered, sample, full=True)
            if score > MIN_SSIM_SCORE:
                header.ssim_scores.append(SsimScore(score, s_id))

    if header.ssim_scores:
        header.ssim_scores.sort(key=lambda k: -k.score)
        header.value = header.ssim_scores[0].value
    return header


def check_headers(col_headers, row_headers):
    check_header(col_headers, True, True)
    check_header(row_headers, False, True)


def check_header(headers, is_row, throw_exception):
    for i in range(GRID_SHAPE[1 - int(is_row)]):
        block_headers = [headers[b][i] for b in range(4)]
        s = sum(b.value for b in block_headers)
        valid_ssims = []
        if GRID_SHAPE[int(is_row)] != s:
            logging.warning(f"Image parsing error: {'Row' if not is_row else 'Column'} Header {i} sums to {s}")
            # test all combinations of all values, keep the best scored valid combination
            h1, h2, h3, h4 = block_headers
            for s1 in h1.ssim_scores:
                for s2 in h2.ssim_scores:
                    for s3 in h3.ssim_scores:
                        for s4 in h4.ssim_scores:
                            s_val = s1.value + s2.value + s3.value + s4.value
                            s_score = s1.score + s2.score + s3.score + s4.score
                            if s_val == GRID_SHAPE[int(is_row)]:
                                valid_ssims.append((s_score, s1, s2, s3, s4))
            if valid_ssims:
                valid_ssims.sort()
                h1.value, h2.value, h3.value, h4.value = (s.value for s in valid_ssims[0][1:])
                return True
            elif throw_exception:
                raise ValueError("Parsing Error")
            return False
    return True


def get_level_image(save_image=False, load_image=True):
    image_level_export_file = '../output/hungry_cat_level.png'
    if load_image:
        img = cv2.imread(image_level_export_file)
    else:
        img = connector.get_screenshot_opencv(pull=False)
        if save_image:
            cv2.imwrite(image_level_export_file, image_level)
    return img


def parse_headers(image):
    col_headers = parse_column_headers(image)
    row_headers = parse_row_headers(image)
    check_headers(col_headers, row_headers)

    # we print in case it's wrong, to be able to load from 'load_headers_from_string'
    print("COLS", col_headers)
    print("ROWS", row_headers)
    return col_headers, row_headers


def load_headers_from_string(col_headers, row_header):
    return load_header_from_string(col_headers, False), load_header_from_string(row_header, True)


def load_header_from_string(header, is_row):
    ret = []
    for b, h in enumerate(header[2:-2].split('], [')):
        brush = []
        ret.append(brush)
        for c, k in enumerate(h.split(', ')):
            circled = k[0] == '('
            i = b if is_row else c
            j = c if is_row else b
            v = int(k[1:-1]) if circled else int(k)
            brush.append(Header(i, j, v, circled, b))
    return ret


if __name__ == "__main__":
    config = config_manager.get_configuration()
    log_manager.initialize_log(config.log_dir, log_level=config.log_level)
    connector = AdbConnector(ip=config.phone_ip, adbkey_path=config.adbkey_path, output_dir=config.output_dir)
    load_samples()
    solver = Solver(connector)

    image_level = get_level_image(save_image=False, load_image=False)
    ch, rh = parse_headers(image_level)
    # chs = '[[6, 5, 4, (2), (3), (2), (3), 5, 8, (8)], [0, 1, (2), 1, 0, 0, 3, 2, 1, 2], [4, 3, 4, (7), (7), (4), 0, 0, 0, 0], [5, 6, 5, 5, 5, 9, 9, 8, 6, 5]]'
    # rhs = '[[0, 0, 1, (2), 4, 5, 4, 5, 5, (6), (6), (3), (2), 1, (2)], [6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (3), 2], [(2), (3), (3), (6), 3, (2), (3), (2), (2), (2), 1, 0, 0, 0, 0], [2, 6, 6, (2), (3), (3), (3), (3), (3), (2), 3, 7, (8), 6, (6)]]'
    # ch, rh = load_headers_from_string(chs, rhs)
    solver.solve_level(ch, rh)
    # solver.push_solution_cell_by_cell()
    solver.push_solution_line_by_line()
