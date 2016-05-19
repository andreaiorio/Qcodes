import multiprocessing as mp
import sys
from datetime import datetime
import time
from traceback import print_exc, format_exc
from uuid import uuid4
import builtins
import logging
import signal

from .helpers import in_notebook

MP_ERR = 'context has already been set'

QUERY_WRITE = 'WRITE'
QUERY_ASK = 'ASK'
RESPONSE_OK = 'OK'
RESPONSE_ERROR = 'ERROR'


def set_mp_method(method, force=False):
    """
    An idempotent wrapper for multiprocessing.set_start_method.

    The most important use of this is to force Windows behavior
    on a Mac or Linux: set_mp_method('spawn')
    args are the same:

    method: one of:
        'fork' (default on unix/mac)
        'spawn' (default, and only option, on windows)
        'forkserver'
    force: allow changing context? default False
        in the original function, even calling the function again
        with the *same* method raises an error, but here we only
        raise the error if you *don't* force *and* the context changes
    """
    try:
        mp.set_start_method(method, force=force)
    except RuntimeError as err:
        if err.args != (MP_ERR, ):
            raise

    mp_method = mp.get_start_method()
    if mp_method != method:
        raise RuntimeError(
            'unexpected multiprocessing method '
            '\'{}\' when trying to set \'{}\''.format(mp_method, method))


class QcodesProcess(mp.Process):

    """
    Modified multiprocessing.Process specialized to Qcodes needs.

    - Nicer repr
    - Automatic streaming of stdout and stderr to our StreamQueue singleton
      for reporting back to the main process
    - Ignore interrupt signals so that commands in the main process can be
      canceled without affecting server and background processes.
    """

    def __init__(self, *args, name='QcodesProcess', queue_streams=True,
                 daemon=True, **kwargs):
        """
        Construct the QcodesProcess but like Process, does not start it.

        name: string to include in repr, and in the StreamQueue
            default 'QcodesProcess'
        queue_streams: should we connect stdout and stderr to the StreamQueue?
            default True
        daemon: should this process be treated as daemonic, so it gets
            terminated with the parent.
            default True, overriding the base inheritance
        any other args and kwargs are passed to multiprocessing.Process
        """
        # make sure the singleton StreamQueue exists
        # prior to launching a new process
        if queue_streams and in_notebook():
            self.stream_queue = get_stream_queue()
        else:
            self.stream_queue = None
        super().__init__(*args, name=name, daemon=daemon, **kwargs)

    def run(self):
        """Executed in the new process, and calls the target function."""
        # ignore interrupt signals, as they come from `KeyboardInterrupt`
        # which we want only to apply to the main process and not the
        # server and background processes (which can be halted in different
        # ways)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self.stream_queue:
            self.stream_queue.connect(str(self.name))
        try:
            super().run()
        except:
            # if we let the system print the exception by itself, sometimes
            # it disconnects the stream partway through printing.
            print_exc()
        finally:
            if (self.stream_queue and
                    self.stream_queue.initial_streams is not None):
                self.stream_queue.disconnect()

    def __repr__(self):
        """Shorter and more helpful repr of our processes."""
        cname = self.__class__.__name__
        r = super().__repr__()
        r = r.replace(cname + '(', '').replace(')>', '>')
        return r.replace(', started daemon', '')


def get_stream_queue():
    """
    Convenience function to get a singleton StreamQueue.

    note that this must be called from the main process before starting any
    subprocesses that will use it, otherwise the subprocess will create its
    own StreamQueue that no other processes know about
    """
    if StreamQueue.instance is None:
        StreamQueue.instance = StreamQueue()
    return StreamQueue.instance


class StreamQueue:

    """
    Manages redirection of child process output for the main process to view.

    Do not instantiate this directly: use get_stream_queue so we only make one.
    One StreamQueue should be created in the consumer process, and passed
    to each child process. In the child, we call StreamQueue.connect with a
    process name that will be unique and meaningful to the user. The consumer
    then periodically calls StreamQueue.get() to read these messages.

    inspired by http://stackoverflow.com/questions/23947281/
    """

    instance = None

    def __init__(self, *args, **kwargs):
        """Create a StreamQueue, passing all args & kwargs to Queue."""
        self.queue = mp.Queue(*args, **kwargs)
        self.last_read_ts = mp.Value('d', time.time())
        self._last_stream = None
        self._on_new_line = True
        self.lock = mp.RLock()
        self.initial_streams = None

    def connect(self, process_name):
        """
        Connect a child process to the StreamQueue.

        After this, stdout and stderr go to a queue rather than being
        printed to a console.

        process_name: a short string that will clearly identify this process
            to the user.
        """
        if self.initial_streams is not None:
            raise RuntimeError('StreamQueue is already connected')

        self.initial_streams = (sys.stdout, sys.stderr)

        sys.stdout = _SQWriter(self, process_name)
        sys.stderr = _SQWriter(self, process_name + ' ERR')

    def disconnect(self):
        """Disconnect a child from the queues and revert stdout & stderr."""
        if self.initial_streams is None:
            raise RuntimeError('StreamQueue is not connected')
        sys.stdout, sys.stderr = self.initial_streams
        self.initial_streams = None

    def get(self):
        """Read new messages from the queue and format them for printing."""
        out = ''
        while not self.queue.empty():
            timestr, stream_name, msg = self.queue.get()
            line_head = '[{} {}] '.format(timestr, stream_name)

            if self._on_new_line:
                out += line_head
            elif stream_name != self._last_stream:
                out += '\n' + line_head

            out += msg[:-1].replace('\n', '\n' + line_head) + msg[-1]

            self._on_new_line = (msg[-1] == '\n')
            self._last_stream = stream_name

        self.last_read_ts.value = time.time()
        return out

    def __del__(self):
        """Tear down the StreamQueue either on the main or a child process."""
        try:
            self.disconnect()
        except:
            pass

        if hasattr(type(self), 'instance'):
            # so nobody else tries to use this dismantled stream queue later
            type(self).instance = None

        if hasattr(self, 'queue'):
            kill_queue(self.queue)
            del self.queue
        if hasattr(self, 'lock'):
            del self.lock


class _SQWriter:
    MIN_READ_TIME = 3

    def __init__(self, stream_queue, stream_name):
        self.queue = stream_queue.queue
        self.last_read_ts = stream_queue.last_read_ts
        self.stream_name = stream_name

    def write(self, msg):
        try:
            if msg:
                msgtuple = (datetime.now().strftime('%H:%M:%S.%f')[:-3],
                            self.stream_name, msg)
                self.queue.put(msgtuple)

                queue_age = time.time() - self.last_read_ts.value
                if queue_age > self.MIN_READ_TIME and msg != '\n':
                    # long time since the queue was read? maybe nobody is
                    # watching it at all - send messages to the terminal too
                    # but they'll still be in the queue if someone DOES look.
                    termstr = '[{} {}] {}'.format(*msgtuple)
                    # we always want a new line this way (so I don't use
                    # end='' in the print) but we don't want an extra if the
                    # caller already included a newline.
                    if termstr[-1] == '\n':
                        termstr = termstr[:-1]
                    try:
                        print(termstr, file=sys.__stdout__)
                    except ValueError:  # pragma: no cover
                        # ValueError: underlying buffer has been detached
                        # this may just occur in testing on Windows, not sure.
                        pass
        except:
            # don't want to get an infinite loop if there's something wrong
            # with the queue - put the regular streams back before handling
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            raise

    def flush(self):
        pass


class ServerManager:

    """
    Creates and communicates with a separate server process.

    Starts a `QcodesProcess`, and on that process it constructs a server
    object of type `server_class`, which should normally be a subclass of
    `BaseServer`. Client processes query the server via:
    `manager.ask(func_name, *args, **kwargs)`: if they want a response or want
        to wait for confirmation that the query has completed
    `manager.write(func_name, *args, **kwargs)`: if they want to continue
        immediately without blocking for the query.
    The server communicates with this manager via two multiprocessing `Queue`s.
    """

    def __init__(self, name, server_class, shared_attrs=None,
                 query_timeout=None):
        """
        Construct the ServerManager and start its server.

        name: the name of the server. Can include .format specs to insert
            all or part of the uuid
        server_class: the class to create within the new process.
            the constructor will be passed arguments:
                query_queue, response_queue, shared_attrs
            and should start an infinite loop watching query_queue and posting
            responses to response_queue.
        shared_attrs: any objects that need to be passed to the server on
            startup, generally objects like Queues that are picklable only for
            inheritance by a new process.
        query_timeout: (default None) the default max time to wait for
            responses
        """
        self._query_queue = mp.Queue()
        self._response_queue = mp.Queue()
        self._server_class = server_class
        self._shared_attrs = shared_attrs

        # query_lock is only used with queries that get responses
        # to make sure the process that asked the question is the one
        # that gets the response.
        # Any query that does NOT expect a response can just dump it in
        # and move on.
        self.query_lock = mp.RLock()

        # uuid is used to pass references to this object around
        # for example, to get it after someone else has sent it to a server
        self.uuid = uuid4().hex

        self.name = name.format(self.uuid)

        self.query_timeout = query_timeout
        self._start_server()

    def _start_server(self):
        self._server = QcodesProcess(target=self._run_server, name=self.name)
        self._server.start()

    def _run_server(self):
        self._server_class(self._query_queue, self._response_queue,
                           self._shared_attrs)

    def _check_alive(self):
        try:
            if not self._server.is_alive():
                logging.warning('restarted {}'.format(self._server))
                self.restart()
        except:
            # can't test is_alive from outside the main process
            pass

    def write(self, func_name, *args, **kwargs):
        """
        Send a query to the server that does not expect a response.

        `write(func_name, *args, **kwargs)` proxies to server method:
        `server.handle_<func_name>(*args, **kwargs)`
        """
        self._check_alive()
        self._query_queue.put((QUERY_WRITE, func_name, args, kwargs))

    def ask(self, func_name, *args, timeout=None, **kwargs):
        """
        Send a query to the server and wait for a response.

        `resp = ask(func_name, *args, **kwargs)` proxies to server method:
        `resp = server.handle_<func_name>(*args, **kwargs)`

        optional timeout (default None) - not recommended, as if we quit
        before reading the response, the query queue can get out of sync
        """
        self._check_alive()

        timeout = timeout or self.query_timeout
        self._expect_error = False

        query = (QUERY_ASK, func_name, args, kwargs)

        with self.query_lock:
            # in case a previous query errored and left something on the
            # response queue, clear it
            while not self._response_queue.empty():
                logging.warning(
                    'unexpected data in response queue before ask:\n' +
                    repr(self._response_queue.get()))

            self._query_queue.put(query)

            res = self._response_queue.get(timeout=timeout)

            while not self._response_queue.empty():
                logging .warning(
                    'unexpected multiple responses in queue during ask, '
                    'using the last one. earlier item(s):\n' +
                    repr(res))
                res = self._response_queue.get(timeout=timeout)

            try:
                code, value = res
            except (TypeError, ValueError):
                code, value = '<MALFORMED>', res

            if code == RESPONSE_OK:
                return value

            self._handle_error(code, value, query)

    def _handle_error(self, code, error_str, query=None):
        error_head = '*** error on {} ***'.format(self.name)

        if query:
            error_head += '\nwhile executing query: {}'.format(repr(query))

        if code != RESPONSE_ERROR:
            error_head += '\nunrecognized response code: {}'.format(code)

        # try to match the error type, if it's a built-in type
        error_type_line = error_str.rstrip().rsplit('\n', 1)[-1]
        error_type_str = error_type_line.split(':')[0].strip()

        err_type = getattr(builtins, error_type_str, None)
        if err_type is None or not issubclass(err_type, Exception):
            err_type = RuntimeError

        raise err_type(error_head + '\n\n' + error_str)

    def halt(self, timeout=2):
        """
        Halt the server and end its process.

        Does not tear down, after this the server can still be started again.
        """
        try:
            if self._server.is_alive():
                self.write('halt')
            self._server.join(timeout)

            if self._server.is_alive():
                self._server.terminate()
                logging.warning('ServerManager did not respond to halt '
                                'signal, terminated')
                self._server.join(timeout)
        except AssertionError:
            # happens when we get here from other than the main process
            # where we shouldn't be able to kill the server anyway
            pass

    def restart(self):
        """Restart the server."""
        self.halt()
        self._start_server()

    def close(self):
        """Irreversibly stop the server and manager."""
        self.halt()
        for q in ['query', 'response', 'error']:
            qname = '_{}_queue'.format(q)
            if hasattr(self, qname):
                kill_queue(getattr(self, qname))
                del self.__dict__[qname]
        if hasattr(self, 'query_lock'):
            del self.query_lock


class BaseServer:

    """
    Base class for servers to run in separate processes.

    The server is started inside a `QcodesProcess` by a `ServerManager`,
    and unifies the query handling protocol so that we are robust against
    deadlocks, out of sync queues, or hidden errors.

    This base class doesn't start the event loop, a subclass should
    either call `self.run_event_loop()` at the end of its `__init__` or
    provide its own event loop. If making your own event loop, be sure to
    call `self.process_query(query)` on any item that arrives in
    `self._query_queue`.

    Subclasses should define handlers `handle_<func_name>`, such that calls:
        `response = server_manager.ask(func_name, *args, **kwargs)`
        `server_manager.write(func_name, *args, **kwargs)`
    map onto method calls:
        `response = self.handle_<func_name>(*args, **kwargs)`

    The actual query passed through the queue and unpacked by `process_query`
    has the form `(code, func_name[, args][, kwargs])` where `code` is:

    - `QUERY_ASK` (from `server_manager.ask`): will always send a response,
      even if the function returns nothing (None) or throws an error.

    - `QUERY_WRITE` (from `server_manager.write`): will NEVER send a response,
      return values are ignored and errors go to the logging framework.

    Two handlers are predefined:

    - `handle_halt` (but override it if your event loop does not use
      self.running=False to stop)

    - `handle_get_handlers` (lists all available handler methods)
    """

    # just for testing - how long to allow it to wait on a queue.get
    # in real situations this should always be None
    timeout = None

    def __init__(self, query_queue, response_queue, shared_attrs=None):
        """
        Create the BaseServer.

        Subclasses should match this call signature exactly, even if they
        do not need shared_attrs, because it is used by `ServerManager`
        to instantiate the server.
        The base class does not start the event loop, subclasses should do
        this at the end of their own `__init__`.

        query_queue: a multiprocessing.Queue that we listen to

        response_queue: a multiprocessing.Queue where we put responses

        shared_attrs: (default None) any objects (such as other Queues)
            that we need to supply on initialization of the server because
            they cannot be picked normally to pass through the Queue later.
        """
        self._query_queue = query_queue
        self._response_queue = response_queue
        self._shared_attrs = shared_attrs

    def run_event_loop(self):
        """
        The default event loop. When this method returns, the server stops.

        Override this method if you need to do more than just process queries
        repeatedly, but make sure your event loop:
        - calls `self.process_query` to ensure robust error handling
        - provides a way to halt the server (and override `handle_halt` if
          it's not by setting `self.running = False`)
        """
        self.running = True
        while self.running:
            query = self._query_queue.get(timeout=self.timeout)
            self.process_query(query)

    def process_query(self, query):
        """
        Act on one query received through the query queue.

        query: should have the form `(code, func_name[, args][, kwargs])`
        """
        try:
            code = None
            code, func_name = query[:2]

            func = getattr(self, 'handle_' + func_name)

            args = None
            kwargs = None
            for part in query[2:]:
                if isinstance(part, tuple) and args is None:
                    args = part
                elif isinstance(part, dict) and kwargs is None:
                    kwargs = part
                else:
                    raise ValueError(part)

            if code == QUERY_ASK:
                self._process_ask(func, args or (), kwargs or {})
            elif code == QUERY_WRITE:
                self._process_write(func, args or (), kwargs or {})
            else:
                raise ValueError(code)
        except:
            self.report_error(query, code)

    def report_error(self, query, code):
        """
        Common error handler for all queries.

        QUERY_ASK puts errors into the response queue for the asker to see.
        QUERY_WRITE shouldn't write a response, so it logs errors instead.
        Unknown modes do *both*, because we don't know where the user will be
        looking and an error that severe it's OK to muck up the queue.
        That's the only way you'll get a response without asking for one.
        """
        error_str = (
            'Expected query to be a tuple (code, func_name[, args][, kwargs]) '
            'where code is QUERY_ASK or QUERY_WRITE, func_name points to a '
            'method `handle_<func_name>`, and optionally args is a tuple and '
            'kwargs is a dict\nquery: ' + repr(query) + '\n' + format_exc())

        if code != QUERY_ASK:
            logging.error(error_str)
        if code != QUERY_WRITE:
            try:
                self._response_queue.put((RESPONSE_ERROR, error_str))
            except:
                logging.error('Could not put error on response queue\n' +
                              error_str)

    def _process_ask(self, func, args, kwargs):
        try:
            response = func(*args, **kwargs)
            self._response_queue.put((RESPONSE_OK, response))
        except:
            self._response_queue.put(
                (RESPONSE_ERROR, repr((func, args, kwargs)) + '\n' +
                    format_exc()))

    def _process_write(self, func, args, kwargs):
        try:
            func(*args, **kwargs)
        except:
            logging.error(repr((func, args, kwargs)) + '\n' + format_exc())

    def handle_halt(self):
        """
        Quit this server.

        Just sets self.running=False, which the default event loop looks for
        between queries. If you provide your own event loop and it does NOT
        look for self.running, you should override this handler with a
        different way to halt.
        """
        self.running = False

    def handle_get_handlers(self):
        """List all available query handlers."""
        handlers = []
        for name in dir(self):
            if name.startswith('handle_') and callable(getattr(self, name)):
                handlers.append(name[len('handle_'):])
        return handlers


def kill_queue(queue):
    """Tear down a multiprocessing.Queue to help garbage collection."""
    try:
        queue.close()
        queue.join_thread()
    except:
        pass
