"""

Author: Tranter Tech
Date: 2024
"""
import socket
import select
import logging
import session

from utils import tools
from enum import Enum
from threading import Thread, Event, Lock
from queue import Queue, Empty
from time import sleep

# globals
_CONFIG = None
_SOCKET_TIMEOUT = 0.01


class SocketMode(Enum):
    LISTEN = 0
    READWRITE = 1


class TCPSocket:
    def __init__(self, mode: SocketMode, config: dict, connection=None):
        """
        Base level TCP socket class for use across the application.
        :param mode: LISTEN or READWRITE enums, listen reserved for server.
        :param config: configuration dictionary.
        :param connection: connection object to override connecting a socket.
        """

        # address parameters
        self.address = config.get('tcp_address', '127.0.0.1')
        self.port = config.get('tcp_port', 7821)

        # if not overriding then create the socket and bind if necessary
        if connection is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if mode == SocketMode.LISTEN:
                # bind to the supplied address
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.bind((self.address, self.port))
                self.socket.listen()
        else:
            # otherwise, override with a connection
            self.socket = connection

        # set the timeout
        self.socket.settimeout(_SOCKET_TIMEOUT)

    def connect(self):
        # connect to the specified address
        self.socket.connect((self.address, self.port))

    def accept(self):
        # accept a connection and return the object and address
        return self.socket.accept()

    def send(self, msg: str) -> int:
        """
        Send method that transmits a string over the socket
        :param msg: message to send
        :return: number of sent bytes
        """

        # get the length and preface the message with how much data to expect
        MSGLEN = len(msg)
        msg = str(MSGLEN).zfill(8) + msg

        # break it into chunks and send the message
        totalsent = 0
        while totalsent < MSGLEN:
            sent = self.socket.send(msg[totalsent:].encode('UTF-8'))
            if sent == 0:
                raise SocketDisconnection("Socket dumped while sending.")
            totalsent = totalsent + sent

        # return what we sent
        return totalsent

    def receive(self, fast=False):
        """
        Method for receiving data over a socket connection
        :param fast: Whether the connection should time out for the default time or a quicker time.
        :return: string that was transmitted.
        """
        # determine if the socket is ready
        if not fast:
            ready = select.select([self.socket], [], [], _SOCKET_TIMEOUT)
        else:
            ready = select.select([self.socket], [], [], 0.1)

        # raise an error
        if not ready[0]:
            raise SocketTimeout('Timed out waiting for length.')

        # determine how much data is coming from the header
        try:
            raw_data = self.socket.recv(8)
        except ConnectionResetError as e:
            raise SocketDisconnection(f'Connection reset: ({e.args}).')
        if raw_data == b'':
            raise SocketDisconnection('Disconnected on read.')

        MSGLEN = int(raw_data.decode('UTF-8'))

        # prepare to get chunks
        chunks = []
        bytes_recd = 0

        # try getting the data
        while bytes_recd < MSGLEN:
            # determine if the socket is ready
            ready = select.select([self.socket], [], [], _SOCKET_TIMEOUT)

            # raise an error
            if not ready[0]:
                raise SocketTimeout('Timed out getting message.')

            chunk = self.socket.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == b'':
                raise SocketDisconnection('Client disconnected during transmission.')
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)

        # return the transmitted and decoded string
        return b''.join(chunks).decode('UTF-8')

    def close(self):
        """
        Close the socket.
        :return:
        """
        self.socket.close()


class SocketTimeout(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)


class SocketDisconnection(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)


class FIFO:
    def __init__(self):
        """
        FIFO class which is used for communications by the server, clients and app. Can be asynchronous when used
        by the respective classes.
        """
        self.host_queue = Queue()
        self.client_queue = Queue()

    def read(self, block=True, timeout=None):
        """
        Read function that can be used by the app on either side of the connection. Blocking is true by default unless
        specified. Errors must be handled by the caller.
        :param block: Block the queue read by default.
        :param timeout: How long to wait when blocking, None by default.
        :return: Data in the queue read from the socket.
        """
        if not block:
            return self.host_queue.get(block=False)
        else:
            return self.host_queue.get(block=True, timeout=timeout)

    def send(self, message):
        """
        Send function that can be used by the app on either side of the connection.
        :param message: Data sent across a socket handled by the server/client connection
        :return: None
        """
        self.client_queue.put(message)

    def data_ready(self):
        """
        Determine whether there is data waiting to be read
        :return: (bool) data is ready to go
        """
        return not self.host_queue.empty()

    def server_read(self):
        """
        Read function for the server. Blocking is set to False for asynchronous operation.
        :return: Data sitting in the queue to be transmitted to the client.
        """
        return self.client_queue.get(False)

    def server_put(self, message):
        """
        Put operation for the server, this is data from the client socket to be given to the host application.
        :param message: Message received from the client
        :return: None
        """
        self.host_queue.put(message)


class ServerConnection:
    def __init__(self, id, conn, fifo):
        """
        Server connection object that is used by the server.
        :param id: Associated connection id
        :param conn: connection accepted by the server
        :param fifo: fifo that is either registered and provided by the server or assigned here
        """
        self.id = id
        self.connection = TCPSocket(SocketMode.READWRITE, {}, conn)
        if fifo is not None:
            self.fifo = fifo
        else:
            self.fifo = FIFO()

        # events which determine whether a connection should be dumped and whether this connection can be picked up
        # by a thread.
        self.busy = Event()
        self.dump = Event()


class ClientConnection:
    def __init__(self, id, fifo, config):
        """
        Class for handling client connections. Will run asynchronously and try to reconnect if the connection drops.
        :param id: connection ID for dealing with the server
        :param fifo: FIFO class to be specified by the parent object
        :param config: config specifying the address, port and other config params
        """
        self.id = id
        self.connection = None  # connection that is defined during the connect function
        self.config = config
        self.fifo = fifo

        # event for halting the comms cycle
        self.halt_comms = Event()
        self.comms_thread = Thread(target=self.communication_thread)
        self.connected = False  # track if we're connected

        # logger for this connection
        self.log = logging.getLogger(f'ClientConn:[{self.id}]')
        self.log.addHandler(session.display_log_handler)

    def connect(self):
        try:
            self.connection = TCPSocket(SocketMode.READWRITE, self.config, None)
            self.log.debug(f'Attempting connection to {self.connection.address}:{self.connection.port} ...')
            self.connection.connect()

            self.connection.send(self.id)
            hs_packet = self.connection.receive()
            if hs_packet == '<SA>':
                self.connection.send('<CA>')
                self.log.info('Handshake with server complete.')
                return True
            else:
                self.log.warning('Server failed auth. Dumping.')
                raise ConnectionRefusedError(f'Handshake failed: auth received - {hs_packet}.')
        except ConnectionRefusedError as e:
            self.log.warning(f'Connection refused. {e.args}')
            sleep(1.0)
            return False
        except SocketDisconnection as e:
            self.log.warning(f'Socket disconnected. {e.args}')
            sleep(1.0)
            return False
        except SocketTimeout as e:
            self.log.warning(f'Socket timed out. {e.args}')
            sleep(1.0)
            return False

    def communication_thread(self):
        while not self.halt_comms.is_set():
            # if we are not connected then let's connect
            if not self.connected:
                self.connected = self.connect()
            else:
                try:
                    data = self.connection.receive(fast=True)
                    self.fifo.server_put(data)
                except SocketTimeout as e:
                    # ignore a timeout
                    pass
                except SocketDisconnection as e:
                    self.log.warning(f'Socket disconnected [{self.id}]. Dumping.')
                    self.connected = False
                    continue

                try:
                    data = self.fifo.server_read()
                    bytes_sent = self.connection.send(data)
                except Empty as e:
                    # if the queue is empty just ignore it
                    pass
                except SocketDisconnection as e:
                    self.log.warning(f'Socket disconnected [{self.id}]. Dumping.')
                    self.connected = False

        # close the connection if we need to
        if self.connected:
            self.connection.close()

        self.log.info('Communication thread exited.')

    def close(self):
        self.halt_comms.set()
        self.comms_thread.join()


class Server:
    def __init__(self, config):
        """
        Server class for handling communications across the application
        :param config: config dictionary, loaded from the parent class
        """
        self.config = config

        # server socket for accepting connections
        self.server_socket = TCPSocket(SocketMode.LISTEN, config)

        # list of connections to manage
        self.connection_lock = Lock()
        self.connections = []
        self.connection_threads = []

        # FIFOs registered byt the app to be used by connections
        self.registered_fifos = {}

        # listening thread
        self.halt_listening = Event()
        self.listening_thread = Thread(target=self._accept_connections)

        # event for the communication threads
        self.halt_comms = Event()

        # dumping thread
        self.dumping_thread = Thread(target=self._dump_thread)

        # get the logger
        self.log = logging.getLogger('Server')
        self.log.addHandler(session.display_log_handler)

    def start_listening(self):
        """
        Start the listening and dumping threads to run asynchronously.
        :return: None
        """
        self.log.info(f"Server started @{self.config.get('tcp_address', None)}"
                      f":{self.config.get('tcp_port', None)}")
        self.listening_thread.start()
        self.dumping_thread.start()

    def _handshake(self, connection):
        """
        Facilitate the handshake between the server and a connection supplied by the accept connections function.
        :param connection: Socket object after accepting
        :return: (connected (bool), id (str))
        """
        conn = TCPSocket(SocketMode.READWRITE, self.config, connection=connection)
        self.log.debug('Waiting to receive ID...')
        fail_state = (False, '')
        try:
            # get the ID from the connection
            id_string = conn.receive()
            self.log.debug(f'Found ID: [{id_string}]. Sending handshake...')

            # acknowledge the receipt
            conn.send('<SA>')
            self.log.debug(f'Sever acknowledge sent. Waiting for <CA>...')

            # acknowledgment by the connection
            conn_ack = conn.receive()
            if conn_ack != '<CA>':
                self.log.debug(f'Expected <CA>, got {conn_ack}. Handshake Failed.')
                return fail_state

            # return it all
            self.log.debug(f'Got <CA>. Handshake complete.')
            return True, id_string
        except SocketTimeout:
            self.log.warning('Connection timed out during handshake. Dumping connection.')
            conn.close()
            return fail_state
        except SocketDisconnection:
            self.log.warning('Connection closed during handshake. Dumping connection.')
            return fail_state

    def _accept_connections(self):
        """
        Loop for accepting connections and adding them to the connection list
        :return:
        """
        while not self.halt_listening.is_set():
            try:
                # accept the connection and perform the handshake
                conn, addr = self.server_socket.accept()
                self.log.debug(f'Connection found: {addr}')
                complete, id = self._handshake(conn)

                # after completing the handshake add it to the connections
                if complete:
                    self.connection_lock.acquire()

                    # find a fifo to pass to the server connection
                    # NOTE: these should be predefined by the application so as not to confuse connections later
                    fifo = self.get_FIFO(id)
                    self.connections.append(ServerConnection(id, conn, fifo))

                    # start a thread to watch the connection
                    new_thread = Thread(target=self._communication_thread)
                    self.connection_threads.append(new_thread)
                    new_thread.start()

                    self.log.debug(f'Connection added. Current conns: ({len(self.connections)})')
                    self.connection_lock.release()
            except TimeoutError:
                pass

    def register_FIFO(self, conn_id: str, fifo: FIFO):
        """
        Register a FIFO object to the sever that will be assigned by the server upon handshaking.
        :param conn_id: connection ID to be used for specifying the connection
        :param fifo: FIFO object instance to be used by the server
        :return: None
        """
        self.log.debug(f'Registering FIFO for [{conn_id}].')
        self.registered_fifos[conn_id.lower()] = fifo

    def get_FIFO(self, conn_id: str):
        """
        Find a FIFO for a matching connection ID
        :param conn_id: (str) connection ID
        :return: the FIFO object to be used
        """
        return self.registered_fifos.get(conn_id.lower(), None)

    def _communication_thread(self):
        """
        Looping communication cycle that handles thread assignment and communications
        :return: (int) exit code
        """
        server_conn = None
        assignment_count = 0    # track the assignment attempts
        while not self.halt_comms.is_set():
            # if we have no object to talk to yet, find one
            if server_conn is None:
                # grab the lock when accessing the connections list
                self.connection_lock.acquire()

                # loop over them and find a non-busy connection
                for sc in self.connections:
                    if not sc.busy.is_set():
                        server_conn = sc
                        server_conn.busy.set()
                        self.log.debug(f'Thread assigned to connection [{sc.id}].')
                        break

                # release the lock
                self.connection_lock.release()

                if server_conn is None:
                    if assignment_count < 3:
                        assignment_count += 1
                        self.log.debug(f'No free server conn found ({assignment_count}).')
                        sleep(1)
                    else:
                        self.log.debug('No free server conn found. Exiting.')
                        return 1
            else:
                # attempt to read
                try:
                    data = server_conn.connection.receive()
                    server_conn.fifo.server_put(data)
                except SocketTimeout as e:
                    # ignore a timeout
                    pass
                except SocketDisconnection as e:
                    self.log.warning(f'Socket disconnected [{server_conn.id}]. Dumping.')
                    server_conn.dump.set()
                    return 1

                # attempt to send
                try:
                    data = server_conn.fifo.server_read()
                    bytes_sent = server_conn.connection.send(data)
                except Empty as e:
                    # if the queue is empty just ignore it
                    pass
                except SocketDisconnection as e:
                    self.log.warning(f'Socket disconnected [{server_conn.id}]. Dumping.')
                    server_conn.dump.set()
                    return 1

        # exit successfully
        return 0

    def _dump_thread(self):
        """
        Go through and dump connections and threads that need to be cleaned up
        :return: None
        """
        self.log.debug('Dumping thread started.')
        while not self.halt_comms.is_set():
            # find all the connections that should be dumped
            dump_conns_idx = []
            self.connection_lock.acquire()
            for idx, server_conn in enumerate(self.connections):
                if server_conn.dump.is_set():
                    dump_conns_idx.append(idx)

            # pop them from the connection list
            dump_conns_idx.reverse()
            dump_conns = []
            for idx in dump_conns_idx:
                dump_conns.append(self.connections.pop(idx))

            self.connection_lock.release()

            # now they're out of the connection list close and delete them
            for sc in dump_conns:
                sc.connection.close()
                self.log.debug(f'Dumped [{sc.id}]...')
                del sc

            # limit the rep rate
            sleep(0.1)

    def close(self):
        """
        Close the server.
        :return: None
        """
        # send the quit command
        for key, fifo in self.registered_fifos.items():
            self.log.debug(f'Sending quit to [{key}]')
            fifo.send('<QUIT>')

        # stop listening
        self.halt_listening.set()
        if self.listening_thread.is_alive():
            self.listening_thread.join()

        # turn off all the conns
        self.halt_comms.set()
        if self.dumping_thread.is_alive():
            self.dumping_thread.join()

        # wait for conn threads to stop
        for thread in self.connection_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        # close all connections
        self.log.debug('Server closing connections.')
        for server_conn in self.connections:
            self.log.debug(f'Closing [{server_conn.id}]...')
            server_conn.connection.close()

        self.log.debug('Server closed.')


def _test_server():
    global _CONFIG
    _CONFIG = tools.load_config(logging.getLogger('TCP'))

    # create the server socket
    server = Server(_CONFIG)
    server.start_listening()

    manager_fifo = FIFO()
    server.register_FIFO('manager', manager_fifo)

    return server

def _test_client():
    global _CONFIG
    _CONFIG = tools.load_config(logging.getLogger('TCP'))

    # code to start the client connection
    fifo = FIFO()
    client_connection = ClientConnection('manager', fifo, _CONFIG)
    client_connection.comms_thread.start()

    x = input('>')

    # code to close the connection
    client_connection.close()


if __name__ == "__main__":
    _test_client()