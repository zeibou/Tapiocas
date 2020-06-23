# coordinates for 1080 x 2160 screen, depending on grid size / # of brushes


class GridShape:
    def __init__(self, nb_cols, nb_rows, nb_brushes):
        self.nb_cols = nb_cols
        self.nb_rows = nb_rows
        self.nb_brushes = nb_brushes

        # solution cells approx positions:
        self.grid_left = -1
        self.grid_top = -1
        self.grid_right = -1
        self.grid_bottom = -1

        # header cells positions (we need precision for better image comparison, and cells are not linearly spaced)
        self.column_headers_x = [-1] * nb_cols
        self.column_headers_y = [-1] * nb_brushes
        self.row_headers_x = [-1] * nb_brushes
        self.row_headers_y = [-1] * nb_rows
        self.header_width = 50
        self.header_height = 50


class Grid10By15b4(GridShape):
    def __init__(self):
        super().__init__(10, 15, 4)

        self.grid_left = 290
        self.grid_top = 623
        self.grid_right = 1050
        self.grid_bottom = 1760
        self.column_headers_x = [305, 381, 459, 535, 612, 682, 759, 836, 913, 989]
        self.column_headers_y = [381, 444, 508, 571]
        self.row_headers_x = [48, 111, 175, 238]
        self.row_headers_y = [638, 714, 791, 868, 944, 1015, 1092, 1168, 1246, 1322, 1392, 1469, 1546, 1623, 1700]


class Grid10By10b4(GridShape):
    def __init__(self):
        super().__init__(10, 10, 4)

        self.grid_left = 290
        self.grid_top = 810
        self.grid_right = 1050
        self.grid_bottom = 1570
        self.column_headers_x = [304, 380, 458, 534, 611, 681, 758, 835, 912, 988]
        self.column_headers_y = [570, 633, 696, 760]
        self.row_headers_x = [48, 111, 175, 238]
        self.row_headers_y = [826, 903, 980, 1056, 1133, 1204, 1280, 1357, 1434, 1511]


# less than 4 brushes are possible, we just pretend there are 4 colours and some of them have only zeroes
class Grid10By10b3(GridShape):
    def __init__(self):
        super().__init__(10, 10, 3)

        self.grid_left = 290
        self.grid_top = 810
        self.grid_right = 1050
        self.grid_bottom = 1570
        self.column_headers_x = [304, 381, 458, 534, 611, 681, 758, 835, 912, 988]
        self.column_headers_y = [825, 593, 672, 752]  # 825 will be the first grid row (zeroes for first colour)
        self.row_headers_x = [304, 72, 151, 230]  # 304 is first empty grid column
        self.row_headers_y = [826, 903, 980, 1056, 1133, 1204, 1280, 1357, 1434, 1511]


class Grid10By10b2(GridShape):
    def __init__(self):
        super().__init__(10, 10, 2)

        self.grid_left = 290
        self.grid_top = 810
        self.grid_right = 1050
        self.grid_bottom = 1570
        self.column_headers_x = [304, 381, 458, 534, 611, 681, 758, 835, 912, 988]
        self.column_headers_y = [825, 825, 633, 739]  # 825 will be the first grid row (zeroes for first colour)
        self.row_headers_x = [304, 304, 111, 216]  # 304 is first empty grid column
        self.row_headers_y = [826, 903, 980, 1056, 1133, 1204, 1280, 1357, 1434, 1511]
