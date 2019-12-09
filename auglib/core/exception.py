import sys
import warnings
import traceback
from functools import wraps
import ctypes

from .api import lib


class LibraryExceptionWarning(UserWarning):
    pass


class LibraryException(Exception):
    pass


class ExceptionHandling:
    r"""Exception handling strategies"""
    SILENT = 'silent'
    WARNING = 'warning'
    STACKTRACE = 'stacktrace'
    EXCEPTION = 'exception'


_exception_handling = ExceptionHandling.EXCEPTION


def set_exception_handling(strategy: ExceptionHandling):
    r"""Set strategy how to handle exceptions thrown by the c library.

    * ``'silent'``: ignore (shows no message)
    * ``'warning'``: show warning message
    * ``'stacktrace'``: show warning message with stacktrace
    * ``'exception'``: raise an exception

    Args:
        strategy: exception handling strategy

    """
    global _exception_handling
    _exception_handling = strategy


def _check_exception(max_msg_len: int = 2048):
    buffer = ctypes.create_string_buffer(max_msg_len)
    if lib.auglib_check_exception(buffer, max_msg_len):
        msg = buffer.value.decode('ascii')
        global _exception_handling
        if _exception_handling == ExceptionHandling.EXCEPTION:
            raise LibraryException(msg)
        else:
            if _exception_handling == ExceptionHandling.SILENT:
                pass
            elif _exception_handling == ExceptionHandling.WARNING:
                warnings.warn(LibraryExceptionWarning(msg))
            elif _exception_handling == ExceptionHandling.STACKTRACE:
                print('Traceback (most recent call last):', file=sys.stderr)
                traceback.print_stack()
                warnings.warn(LibraryExceptionWarning(msg))
            lib.auglib_release_exception()


def _check_exception_decorator(func):
    # Preserve docstring, see:
    # https://docs.python.org/3.6/library/functools.html#functools.wraps
    @wraps(func)
    def inner(*args, **kwargs):
        res = func(*args, **kwargs)
        _check_exception()
        return res
    return inner
