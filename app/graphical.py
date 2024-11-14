"""

Author: Tranter Tech
Date: 2024
"""
import pyqtgraph as pg
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty as QueueEmpty
from threading import Thread, Event
from pyqtgraph.Qt import QtCore

class PlottingWindow:
    def __init__(self, img_queue):
        self.app = None
        self.win = None
        self.layout = None
        self.imv = None
        self.img_queue = img_queue
        # self.update_thread = Thread(target=self._update_plot)
        self._halt_event = Event()

    def initialise(self):
        self.app = pg.mkQApp("ImageView Example")

        self.win = pg.GraphicsLayoutWidget()

        self.app.aboutToQuit.connect(self._close)

        self.win.show()
        self.win.resize(800, 800)

        view = self.win.addViewBox()
        view.setAspectLocked(True)
        # next_item = self.win.addViewBox()

        self.imv = pg.ImageItem()
        plot = pg.PlotItem()
        view.addItem(self.imv)

        self.win.addItem(plot)

        X = np.linspace(0, np.pi, 1000)
        Y = np.sin(X)
        plot.plot(X, Y)

        # Kinesis, LVL 7, Sem 3

        self.win.setWindowTitle('pyqtgraph example: ImageView')

        # self.update_thread.start()
        # self.timer.start(1000)

        self.timer = pg.QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(100)

        self.run()

    def run(self):
        pg.exec()

    def _close(self):
        self._halt_event.set()
        print('Closing!')

    def _update_plot(self):
        while not self._halt_event.is_set():
            self.imv.setImage(np.random.uniform(0, 255, (50, 50)))


class RealTimePlot():
    def __init__(self, format_string, queue, append=False, refresh_wait=50, title=''):
        # # the plotting process
        self.plot_proc = Process(target=self.start, args=())

        # the place where data will arrive
        self.data_queue = queue

        # the application instance
        self.app = pg.mkQApp("Plotting Example")

        self.title = title
        self.format_string = format_string

        # timer for updating the plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.get_update)
        self.timer.start(50)

        # plotting window
        self.win = pg.GraphicsLayoutWidget()
        self.win.resize(1000, 600)
        #
        # set antialias to be true
        pg.setConfigOptions(antialias=True)

        ## Format for the plotting:
        #   \n          - for new row
        #   ,           - for new column
        #   p{name|title}     - plot with a given key 'name' and a title (optional)

        # curve dict
        self.curves = {}

        # get the rows and max column span
        rows = self.format_string.split('\n')
        max_columns = max([len(x.split('{')) - 1 for x in rows])
        for r in rows:
            # add a row if we need to
            if rows.index(r) != 0:
                self.win.nextRow()

            # find the columns
            columns = r.split(',')
            for c in columns:

                # figure out the respective column span
                col_span = 1
                if len(columns) == 1:
                    col_span = max_columns

                # get the name and set the plots appropriately
                name = c[2:-1].split('|')
                if len(name) > 1:
                    self.curves[name[0]] = [{'x': [], 'y': []},
                                            self.win.addPlot(title=name[1],
                                                             colspan=col_span).plot()
                                            ]
                else:
                    self.curves[name[0]] = [{'x': [], 'y': []},
                                            self.win.addPlot(colspan=col_span).plot()
                                            ]

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

            if mode == 'append':
                new_x = data_dict.get('x', None)
                new_y = data_dict.get('y', [])


                if new_x is None:
                    new_x = [i + len(self.curves[plot][0]['y']) for i in range(len(new_y))]

                self.curves[plot][0]['x'] += list(new_x)
                self.curves[plot][0]['y'] += list(new_y)

                # set the new data
                self.curves[plot][1].setData(self.curves[plot][0]['x'], self.curves[plot][0]['y'],
                                            symbolBrush=args.get('symbolBrush', None),
                                            pen=args.get('pen', 'w'),
                                            symbolPen=args.get('symbolPen', None),
                                             )
                # self.curves[plot][1].setPen(None)
            elif mode == 'replace':
                # get the new data and replace it
                new_x = data_dict.get('x', None)
                new_y = data_dict.get('y', [])

                if new_x is None:
                    new_x = [i + len(self.curves[plot][0]['y']) for i in range(len(new_y))]

                self.curves[plot][0]['x'] = list(new_x)
                self.curves[plot][0]['y'] = list(new_y)

                # set the new data
                self.curves[plot][1].setData(self.curves[plot][0]['x'], self.curves[plot][0]['y'],
                                             symbolBrush=args.get('symbolBrush', None),
                                             pen=args.get('pen', 'w'),
                                             symbolPen=args.get('symbolPen', None),
                                             )
            else:
                print("'%s' is not a valid command for data handling." % mode)
        except QueueEmpty:
            pass

    def get_queue(self):
        return self.data_queue


if __name__ == '__main__':
    img_queue = Queue()
    pw = PlottingWindow(img_queue)
    proc = Process(target=pw.initialise)
    proc.start()
    proc.join()


