"""

Author: Tranter Tech
Date: 2024
"""
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread, Event
from queue import Queue, Empty, Full


class LiveViewCanvas(tk.Canvas):
    def __init__(self, parent, image_queue):
        self.image_queue = image_queue
        self._image_width = 0
        self._image_height = 0
        tk.Canvas.__init__(self, parent)
        self.pack()
        self._get_image()

    def _get_image(self):
        try:
            image = self.image_queue.get_nowait()
            self._image = ImageTk.PhotoImage(master=self, image=image)
            if (self._image.width() != self._image_width) or (self._image.height() != self._image_height):
                # resize the canvas to match the new image size
                self._image_width = self._image.width()
                self._image_height = self._image.height()
                self.config(width=self._image_width, height=self._image_height)
            self.create_image(0, 0, image=self._image, anchor='nw')
        except Empty:
            pass
        self.after(10, self._get_image)


class ImageAcquisitionThread(Thread):
    def __init__(self, camera):
        super(ImageAcquisitionThread, self).__init__()
        self._camera = camera
        self._previous_timestamp = 0

        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0  # Do not want to block for long periods of time
        self._image_queue = Queue(maxsize=2)
        self._stop_event = Event()

        # TODO: aaron code
        self.roi = (0, 1000, 500, 1500)
        self.gain = 3
        self.bit_shift = 12

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_image(self, frame):
        # no coloring, just scale down image to 8 bpp and place into PIL Image object
        scaled_image = (frame.image_buffer >> (self._bit_depth - self.bit_shift)) * self.gain
        xmin, xmax, ymin, ymax = self.roi
        scale = 0.25
        new_x = int((xmax - xmin) * scale)
        new_y = int((ymax - ymin) * scale)
        img = Image.fromarray(scaled_image[xmin:xmax, ymin:ymax])
        try:
            if scale != 1.0:
                img = img.resize((new_x, new_y))
        except Exception as e:
            print(e.args)

        return img

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    pil_image = self._get_image(frame)
                    self._image_queue.put_nowait(pil_image)
            except Full:
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                print("Encountered error: {error}, image acquisition will stop.".format(error=error))
                break
        print("Image acquisition has stopped")