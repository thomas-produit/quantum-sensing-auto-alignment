"""

Author: Tranter Tech
Date: 2024
"""
import pyqtgraph as pg
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty
from pyqtgraph.Qt import QtCore

_DEFAULT_CONFIG = [{'fringe_img': {'type': 'image', 'title': 'Raw Fringe', 'span': 1},
                    'dark_img': {'type': 'image', 'title': 'Dark', 'span': 1},
                    'sub_img': {'type': 'image', 'title': 'Subtracted', 'span': 1}},
                   {'cost_hist': {'type': 'plot', 'title': 'Cost Function'}},
                   {'pred': {'type': 'plot', 'title': 'Predicted vs Actual cost', 'span': 1},
                    'params': {'type': 'plot', 'title': 'Training Parameters', 'span': 1},
                    'dist': {'type': 'plot', 'title': 'Distance from Best', 'span': 1}},
                   {'nets': {'type': 'plot', 'title': 'Current Parameters', 'span': 1},
                    'best': {'type': 'plot', 'title': 'Best Parameters', 'span': 1},
                    'dist_0': {'type': 'plot', 'title': 'Distance from 0', 'span': 1}}
                   ]


class RealTimePlot:
    def __init__(self, format_dict, queue, refresh_wait=50, title='Real time plot', scale=1.0):
        # # the plotting process
        self.plot_proc = Process(target=self.start, args=())

        # the place where data will arrive
        self.data_queue = queue

        # the application instance
        self.app = pg.mkQApp(title)
        self.title = title
        self.format_dict = format_dict

        # plotting window
        self.win = pg.GraphicsLayoutWidget()
        size = int(1000*scale)
        self.win.resize(size, size)

        # set antialias to be true
        pg.setConfigOptions(antialias=True)

        # curve dict
        self.curves = {}

        # get the rows and max column span
        max_columns = max([len(row.keys()) for row in format_dict])

        # loop over all rows
        for row_idx, row_dict in enumerate(format_dict):
            # add a new row after the first row
            if row_idx > 0:
                self.win.nextRow()

            # construct each plot
            for plot, options in row_dict.items():
                # pull out plot options
                plot_type = options.get('type', 'plot')
                title = options.get('title', '')
                span = options.get('span', max_columns)

                if plot_type == 'plot':
                    # add the plot
                    self.curves[plot] = [{'x': [], 'y': [], 'type': plot_type},
                                         self.win.addPlot(title=title, colspan=span).plot()]
                elif plot_type == 'image':
                    # add the viewbox first
                    view = self.win.addViewBox(colspan=span)
                    view.setAspectLocked(True)

                    self.imv = pg.ImageItem(title=title)
                    view.addItem(self.imv)

                    self.curves[plot] = [{'x': [], 'y': [], 'type': plot_type}, self.imv]

        # timer for updating the plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.get_update)
        self.timer.start(refresh_wait)

        self.start()

    def start(self):
        self.win.show()
        pg.exec()

    def run(self):
        self.plot_proc.start()

    def get_update(self):
        try:
            # get the new data and concatenate it
            mode, data_dict, plot = self.data_queue.get(False)

            args = data_dict.get('args', {})
            options, plot_item = self.curves.get(plot, (None, None))

            # ignore a call to a plot that doesn't exist
            if plot_item is None:
                return 1

            if mode == 'append':
                if options.get('type') == 'image':
                    # appending images not supported
                    return 1

                # get the data from the dictionary
                new_x = data_dict.get('x', None)
                new_y = data_dict.get('y', [])

                # run the x-axis longer
                if new_x is None:
                    new_x = [i + len(self.curves[plot][0]['y']) for i in range(len(new_y))]

                # update the data
                options['x'] += list(new_x)
                options['y'] += list(new_y)

                # set the new data
                plot_item.setData(options['x'], options['y'],
                                  symbolBrush=args.get('symbolBrush', None),
                                  pen=args.get('pen', 'w'),
                                  symbolPen=args.get('symbolPen', None))

            elif mode == 'replace':
                # get the new data and replace it
                new_x = data_dict.get('x', None)
                new_y = data_dict.get('y', [])

                options, plot_item = self.curves.get(plot, (None, None))

                # dump if we don't exist
                if plot_item is None:
                    return 1

                if new_x is None and options['type'] == 'plot':
                    new_x = [i + len(self.curves[plot][0]['y']) for i in range(len(new_y))]
                    options['x'] = list(new_x)

                # always get y data, but only x for the plot type
                options['y'] = list(new_y)

                # set the new data
                if options['type'] == 'plot':
                    plot_item.setData(options['x'], options['y'],
                                      symbolBrush=args.get('symbolBrush', None),
                                      pen=args.get('pen', 'w'),
                                      symbolPen=args.get('symbolPen', None))
                elif options['type'] == 'image':
                    img_data = np.array(options['y']).squeeze()
                    plot_item.setImage(img_data)
            else:
                print("'%s' is not a valid command for data handling." % mode)
        except QueueEmpty:
            pass

    def get_queue(self):
        return self.data_queue


if __name__ == '__main__':
    _plot_queue = Queue()
    _graphical_proc = Process(target=RealTimePlot, args=(_DEFAULT_CONFIG, _plot_queue))
    _graphical_proc.start()

    for i in range(100):
        _plot_queue.put(('append', {'y': [np.random.rand()]}, 'cost_hist'))
        _plot_queue.put(('replace', {'y': [np.random.uniform(0, 255, (50, 50))]}, 'fringe_img'))

        img_space = np.repeat(np.linspace(0, np.pi * 2, 50).reshape(1, -1), 50, 0)

        _plot_queue.put(('replace', {'y': np.sin(img_space + i / 10)}, 'dark_img'))
        _plot_queue.put(('replace', {'y': np.cos(img_space + i / 10)}, 'sub_img'))
        _plot_queue.put(('replace', {'y': np.random.uniform(0, 255, 4),
                                     'args': {'pen': None, 'symbolBrush': 'b'}}, 'best'))

    _graphical_proc.join()
