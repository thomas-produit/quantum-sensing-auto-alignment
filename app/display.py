import curses
from . import commands
from queue import Queue, Empty
from threading import Event, Thread, Lock
import session

ok_clr = lambda ret_str: '\033[32m' + ret_str + '\033[0m'
fail_clr = lambda ret_str: '\033[93m' + ret_str + '\033[0m'

class MainDisplay:
    """
    Main display class which is used by the manager to run an interactive session. The display is threaded so that
    commands can be run separate of the main processes.
    """
    def __init__(self):
        self.input_queue = Queue()
        self.command_parser_halt = Event()
        self.refresh_output_event = Event()
        self.cmd_thread = Thread(target=self.command_parser)

        # input buffer captures when the user is typing
        self.input_buffer = []

        # this holds the list of outputs in the program, it will be trimmed after 2000 lines
        self.output_buffer = []
        self.output_colour_buffer = []
        self.output_offset = 0

        # Session object, used to pass info to the command classes
        self.session = session

        self.return_values = [-1, -1]

    # target for an external thread
    def start_display(self):
        """
        Start the curses display by instantiating the main loop. This should be the main entry point for the display.
        :return: None
        """
        curses.wrapper(self.main_display_loop)

    @staticmethod
    def build_window(stdscr):
        """
        Static method which builds the curses window for displaying.
        :param stdscr: Screen object passed from the curses library.
        :return: None
        """
        # clear the screen
        stdscr.clear()

        # get the terminal size
        lines, cols = stdscr.getmaxyx()

        stdscr.addstr(lines - 2, 0, '_' * cols)
        stdscr.addstr('>')

    def refresh_output(self, stdscr):
        """
        Refresh the screen output according to the internal buffer and cursor position.
        :param stdscr: Standard screen element passed from curses.
        :return: None
        """
        # get the terminal size
        lines, cols = stdscr.getmaxyx()
        lines -= 2  # account for the bottom bit
        session.display_width = cols

        # handle no colours
        if self.output_colour_buffer is None:
            self.output_colour_buffer = [1] * len(self.output_buffer)

        # figure out where the 'window' should sit
        N = len(self.output_buffer)
        default_offset = max(0, N - lines)
        display_log = self.output_buffer[default_offset - self.output_offset:N - self.output_offset]
        display_colours = self.output_colour_buffer[default_offset - self.output_offset:N - self.output_offset]

        # reverse the list so the line coordinates make sense
        display_log.reverse()
        display_colours.reverse()
        line_coord = lines - 1

        # write the output buffer to the 'window' region
        for line, color in zip(display_log, display_colours):
            stdscr.addstr(line_coord, 0, line[:cols].ljust(cols, ' '), curses.color_pair(color))
            line_coord -= 1

        # put the cursor back
        input_text = ''.join([chr(x) for x in self.input_buffer])
        stdscr.addstr(stdscr.getmaxyx()[0] - 2 + 1, 1, input_text.ljust(cols - 2, ' '))
        stdscr.move(stdscr.getmaxyx()[0] - 1, len(input_text) + 1)

        # return the new cursor position
        return [stdscr.getmaxyx()[0] - 1, len(input_text) + 1]

    def main_display_loop(self, stdscr):
        """
        Main display loop which is instantiated by the start_display method. This method is in charge of hadnling the
        buffer, key presses and other user interactions. Additionally, the command thread is instantiated here.
        :param stdscr: Standard screen element passed from curses.
        :return: None
        """
        # start the cmd thread
        self.cmd_thread.start()

        # build the main window
        self.build_window(stdscr)

        stdscr.nodelay(1)

        # initialise colours
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)

        # refresh the output and get the positions
        self.refresh_output(stdscr)
        position = list(stdscr.getyx())

        # main interaction loop
        while True:
            buffered_vals = (position, self.output_offset, tuple(self.input_buffer))

            c = stdscr.getch()

            # move around the screen as needed
            if c == curses.KEY_LEFT:
                position[1] -= 1
            elif c == curses.KEY_RIGHT:
                position[1] += 1
            elif c == curses.KEY_UP:
                position[0] -= 1
            elif c == curses.KEY_DOWN:
                position[0] += 1
            elif c == curses.KEY_PPAGE:
                self.output_offset += 1
            elif c == curses.KEY_NPAGE:
                self.output_offset -= 1
            elif 32 <= c < 127:
                self.input_buffer.append(c)
            elif c == 127:
                # handle delete key
                ypos, xpos = stdscr.getyx()
                xpos = clip([xpos], [(0, len(self.input_buffer) - 1)])[0]
                if len(self.input_buffer) > 0:
                    self.input_buffer.pop(xpos)
            elif c == 410:
                # handle resizing the terminal
                self.build_window(stdscr)
                position = self.refresh_output(stdscr)
            elif c == 10:
                # get the text and add it to the
                input_text = ''.join([chr(x) for x in self.input_buffer])

                if input_text == 'q()'.lower():
                    self.return_values[0] = 0
                    self.halt()
                    return 0

                if len(input_text) > 0:
                    self.input_queue.put(input_text)

                self.input_buffer = []

                position = self.refresh_output(stdscr)
            elif c == 4:
                # catch the ctrl+d event and exit
                self.return_values[0] = 0
                self.halt()
                return 0
            elif c == -1:
                # when nothing is pressed, but we fall through, check to see if there is something
                # we should be doing instead like refreshing
                if self.refresh_output_event.is_set():
                    self.refresh_output_event.clear()
                    self.refresh_output(stdscr)
                continue

            # move to the new position
            lines, cols = stdscr.getmaxyx()
            position = clip(position, [(0, lines - 1), (0, cols - 1)])
            stdscr.move(*position)

            # handled the offset scrolling change
            self.output_offset = clip([self.output_offset], [(0, len(self.output_buffer) - lines + 2)])[0]
            if buffered_vals[1] != self.output_offset:
                position = self.refresh_output(stdscr)

            # handle the change to input buffer
            if buffered_vals[2] != tuple(self.input_buffer):
                position = self.refresh_output(stdscr)

    def add_to_output(self, text, colour=1, mode='w'):
        """
        Handler function to add text into the output buffer
        :param text: text to be added
        :param colour: colour pair, default=0
        :param mode: mode to use, default='w' for appending to the list, 'a' to add to the last line
        :return: None
        """
        if mode == 'w':
            self.output_buffer.append(text)
            self.output_colour_buffer.append(colour)
        elif mode == 'a':
            self.output_buffer[-1] += text
            # use colour=None to keep the previous colour
            if colour is not None:
                self.output_colour_buffer[-1] = colour

    def halt(self):
        """
        Halt the display in a thread safe manner.
        :return: None
        """
        self.command_parser_halt.set()
        self.cmd_thread.join()

    def external_log(self, text, colour=1, mode='w'):
        """
        Main handler for connecting logging events to the displays internal buffer. Upon receiving a log event the
        screen is refreshed so as to display the log.
        :param text: text to be added
        :param colour: colour pair, default=0
        :param mode: mode to use, default='w' for appending to the list, 'a' to add to the last line
        :return: None
        """
        self.add_to_output(text, colour, mode)
        self.refresh_output_event.set()

    def command_parser(self):
        """
        Command parser thread that runs asynchronously to the rest of the main loop. This can still pass off the input
        queue to handle back and forth interactions.
        :return: error code (int)
        """
        while not self.command_parser_halt.is_set():

            # try to get the input text otherwise loop back around
            try:
                input_text = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            # make sure it's a string
            if type(input_text) is not str:
                raise TypeError(f'Expected type str but instead got type:{type(input_text)}.')

            # put it in the buffer so the user can see it
            input_text.strip('\n')
            if len(input_text) > 0:
                self.add_to_output(f'>{input_text}', 2)

                # set up the refresh output event
                self.refresh_output_event.set()

            # now we go see what we need to be doing next
            # handle the code that needs to be parsed
            cmd_found = False
            for cmd, obj in commands.registered_commands.items():
                if input_text.lower() == cmd.lower():
                    cmd_found = True
                    cmd_class = obj(self.session)
                    while cmd_class.state is not commands.state.FINISHED:
                        cmd_class.run(self.input_queue)
                        log_data, log_modes, log_colour = cmd_class.get_log_data()

                        # get back the log data from the command running
                        for line, mode, colour in zip(log_data, log_modes, log_colour):
                            self.add_to_output(line, colour=colour, mode=mode)
                        self.refresh_output_event.set()

                    self.add_to_output('<<<< ---- >>>>', 1)
                    self.refresh_output_event.set()

            if not cmd_found:
                self.add_to_output('[Err]: Command not found.', 3)
            self.refresh_output_event.set()

        # exit the loop when finished
        self.return_values[1] = 0
        return 0


def clip(inputs, bounds):
    """
    Clip input to bounds such that input never exceeds a boxed off region
    :param inputs: list of coordinates, [x1, x2, ...]
    :param bounds: list of tuples consituting the min and max bounds, [(x1_min, x1_max), (x2_min, x2_max), ...]
    :return: list of clipped coordinates [x1, x2, ...]
    """
    # return the list of coords
    coords = []

    # loop over the variables
    for clip_var, var_bound in zip(inputs, bounds):
        x_min, x_max = var_bound

        # clip to the bounding box
        clip_var = min(clip_var, x_max)
        clip_var = max(clip_var, x_min)

        coords.append(clip_var)

    return coords