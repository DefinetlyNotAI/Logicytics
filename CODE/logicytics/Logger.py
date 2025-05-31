from __future__ import annotations

import inspect
import logging
import os
import re
import time
from datetime import datetime
from typing import Type

import colorlog

from logicytics.Config import DEBUG


class Log:
    """
    A logging class that supports colored output using the colorlog library.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of the Log class is created (Singleton pattern).

        :param cls: The class being instantiated.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: The single instance of the Log class.
        """
        if cls._instance is None:
            cls._instance = super(Log, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: dict = None):
        """
        Initializes the Log class with the given configuration.

        :param config: A dictionary containing configuration options.
        """
        if self._initialized and config is None:
            return
        self._initialized = True
        if config:
            self.reset()
        # log_path_relative variable takes Logger.py full path,
        # goes up twice then joins with ACCESS\\LOGS\\Logicytics.log
        log_path_relative = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                         "ACCESS\\LOGS\\Logicytics.log")
        config = config or {
            "filename": log_path_relative,
            "use_colorlog": True,
            "log_level": "INFO",
            "debug_color": "cyan",
            "info_color": "green",
            "warning_color": "yellow",
            "error_color": "red",
            "critical_color": "bold_red",
            "exception_color": "red",
            "colorlog_fmt_parameters": "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            "truncate_message": True,
            "delete_log": False,
        }
        self.EXCEPTION_LOG_LEVEL = 45
        self.INTERNAL_LOG_LEVEL = 15
        logging.addLevelName(self.EXCEPTION_LOG_LEVEL, "EXCEPTION")
        logging.addLevelName(self.INTERNAL_LOG_LEVEL, "INTERNAL")
        self.color = config.get("use_colorlog", True)
        self.truncate = config.get("truncate_message", True)

        self.filename = config.get("filename", log_path_relative)
        if self.color:
            logger = colorlog.getLogger()
            logger.setLevel(getattr(logging, config["log_level"].upper(), logging.INFO))
            handler = colorlog.StreamHandler()
            log_colors = {
                "INTERNAL": "cyan",
                "DEBUG": config.get("debug_color", "cyan"),
                "INFO": config.get("info_color", "green"),
                "WARNING": config.get("warning_color", "yellow"),
                "ERROR": config.get("error_color", "red"),
                "CRITICAL": config.get("critical_color", "bold_red"),
                "EXCEPTION": config.get("exception_color", "red"),
            }

            formatter = colorlog.ColoredFormatter(
                config.get(
                    "colorlog_fmt_parameters",
                    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
                ),
                log_colors=log_colors,
            )

            handler.setFormatter(formatter)
            logger.addHandler(handler)
            try:
                getattr(logging, config["log_level"].upper())
            except AttributeError as AE:
                self.__internal(
                    f"Log Level {config['log_level']} not found, setting default level to INFO -> {AE}"
                )

        if not os.path.exists(self.filename):
            self.newline()
            self._raw("|     Timestamp     |  LOG Level  |"
                      + " " * 71
                      + "LOG Messages"
                      + " " * 71
                      + "|")
        elif os.path.exists(self.filename) and config.get("delete_log", False):
            with open(self.filename, "w") as f:
                f.write(
                    "|     Timestamp     |  LOG Level  |"
                    + " " * 71
                    + "LOG Messages"
                    + " " * 71
                    + "|"
                    + "\n"
                )
        self.newline()

    @staticmethod
    def reset():
        """
        Resets the logger by removing all existing handlers.
        """
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    @staticmethod
    def __timestamp() -> str:
        """
        Returns the current timestamp as a string.

        :return: Current timestamp in 'YYYY-MM-DD HH:MM:SS' format.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __trunc_message(self, message: str) -> str:
        """
        Pads or truncates the message to fit the log format.

        :param message: The log message to be padded or truncated.
        :return: The padded or truncated message.
        """
        if self.truncate is False:
            return message + " " * (153 - len(message)) + "|"
        return (
            message + " " * (153 - len(message))
            if len(message) < 153
            else message[:150] + "..."
        ) + "|"

    def __internal(self, message):
        """
        Log an internal message exclusively to the console.
        
        Internal messages are used for logging system states or debug information
        that should not be written to log files.
        These messages are only displayed in the console when color logging is enabled.
        
        Parameters:
            message (str): The internal message to be logged.
            If the message is "None" or None, no logging occurs.
        
        Notes:
            - Requires color logging to be enabled
            - Uses a custom internal log level
            - Converts the message to a string before logging
        """
        if self.color and message != "None" and message is not None:
            colorlog.log(self.INTERNAL_LOG_LEVEL, str(message))

    def debug(self, message):
        """
        Logs a debug message.

        :param message: The debug message to be logged.
        """
        if self.color and message != "None" and message is not None:
            colorlog.debug(str(message))

    def _raw(self, message):
        """
        Log a raw message directly to the log file.
        
        This method writes a message directly to the log file without any additional formatting
        or logging levels.

        WARNING: This method is for internal use only! Using it directly can mess up
        your log file format and make it hard to read. Use info(), debug(), or
        other public methods instead.

        Parameters:
            message (str): The raw message to be written to the log file.
        
        Notes:
            - Checks the calling context to warn about non-function calls
            - Handles potential Unicode encoding errors
            - Skips logging if message is None or "None"
            - Writes message with a newline character
            - Logs internal errors if file writing fails
        
        Raises:
            Logs internal errors for Unicode or file writing issues without stopping execution
        """
        frame = inspect.currentframe().f_back
        if frame and frame.f_code.co_name == "<module>":
            self.__internal(
                f"Raw message called from a non-function - This is not recommended"
            )
        # Precompiled regex for ANSI escape codes
        # Remove all ANSI escape sequences in one pass
        message = re.compile(r'\033\[\d+(;\d+)*m').sub('', message)

        if message and message != "None":
            try:
                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(f"{str(message)}\n")
            except (UnicodeDecodeError, UnicodeEncodeError) as UDE:
                self.__internal(
                    f"UnicodeDecodeError: {UDE} - Message: {str(message)}"
                )
            except Exception as E:
                self.__internal(f"Error: {E} - Message: {str(message)}")

    def newline(self):
        """
        Write a newline separator to the log file, creating a visual divider between log entries.
        
        This method writes a formatted horizontal line to the log file using ASCII characters,
        which helps visually separate different sections or log entries.
        The line consists of vertical bars and dashes creating a structured tabular-like separator.
        
        Side Effects:
            Appends a newline separator to the log file specified by `self.filename`.
        """
        with open(self.filename, "a") as f:
            f.write("|" + "-" * 19 + "|" + "-" * 13 + "|" + "-" * 154 + "|" + "\n")

    def info(self, message):
        """
        Logs an info message.

        :param message: The info message to be logged.
        """
        if self.color and message != "None" and message is not None:
            colorlog.info(str(message))
        self._raw(f"[{self.__timestamp()}] > INFO:     | {self.__trunc_message(str(message))}")

    def warning(self, message):
        """
        Logs a warning message.

        :param message: The warning message to be logged.
        """
        if self.color and message != "None" and message is not None:
            colorlog.warning(str(message))
        self._raw(f"[{self.__timestamp()}] > WARNING:  | {self.__trunc_message(str(message))}")

    def error(self, message):
        """
        Logs an error message.

        :param message: The error message to be logged.
        """
        if self.color and message != "None" and message is not None:
            colorlog.error(str(message))
        self._raw(f"[{self.__timestamp()}] > ERROR:    | {self.__trunc_message(str(message))}")

    def critical(self, message):
        """
        Logs a critical message.

        :param message: The critical message to be logged.
        """
        if self.color and message != "None" and message is not None:
            colorlog.critical(str(message))
        self._raw(f"[{self.__timestamp()}] > CRITICAL: | {self.__trunc_message(str(message))}")

    def string(self, message, type: str):
        """
        Logs a message with a specified log type, supporting multiple type aliases.
        
        This method allows logging messages with flexible type specifications,
        mapping aliases to standard log types and handling potential errors in type selection.
        It supports logging with color if enabled.
        
        Parameters:
            message (str): The message to be logged. Skipped if "None" or None.
            type (str): The log type, which can be one of:
                - Standard types: 'debug', 'info', 'warning', 'error', 'critical'
                - Aliases: 'err' (error), 'warn' (warning), 'crit' (critical), 'except' (exception)
        
        Behavior:
            - Converts type to lowercase and maps aliases to standard log types
            - Logs message using the corresponding log method
            - Falls back to debug logging if an invalid type is provided
            - Only logs if color is enabled and message is not "None"
        
        Raises:
            AttributeError: If no matching log method is found (internally handled)
        """
        if self.color and message != "None" and message is not None:
            type_map = {"err": "error", "warn": "warning", "crit": "critical", "except": "exception"}
            type = type_map.get(type.lower(), type)
            try:
                getattr(self, type.lower())(str(message))
            except AttributeError as AE:
                self.__internal(f"A wrong Log Type was called: {type} not found. -> {AE}")
                getattr(self, "Debug".lower())(str(message))

    def exception(self, message, exception_type: Type = Exception):
        """
        Log an exception message and raise the specified exception.
        
        Warning: Not recommended for production use. Prefer Log().error() for logging exceptions.
        
        Args:
            message (str): The exception message to be logged.
            exception_type (Type, optional): The type of exception to raise. Defaults to Exception.
        
        Raises:
            The specified exception type with the provided message.
        
        Note:
            - Only logs the exception if color logging is enabled and message is not None
            - Logs the exception with a timestamp and truncated message
            - Includes both the original message and the exception type in the log
        """
        if self.color and message != "None" and message is not None:
            self._raw(
                f"[{self.__timestamp()}] > EXCEPTION:| {self.__trunc_message(f'{message} -> Exception provoked: {str(exception_type)}')}")
        raise exception_type(message)

    def execution(self, message_log: list[tuple[str, str]]):
        """
        Parse and log multiple messages with their corresponding log types.
        
        This method processes a list of messages, where each message is associated with a specific log type. It is designed for scenarios where multiple log entries need to be processed simultaneously, such as logging script execution results.
        
        Parameters:
            message_log (list[tuple[str, str]]): A list of message entries.
            Each entry is a list containing two elements:
                - First element: The log message (str)
                - Second element: The log type (str)
        
        Behavior:
            - Iterates through the provided message log
            - Logs each message using the specified log type via `self.string()`
            - Logs an internal warning if a message list does not contain exactly two elements
        
        Example:
            log = Log()
            log.parse_execution([
                ['Operation started', 'info'],
                ['Processing data', 'debug'],
                ['Completed successfully', 'info']
            ])
        """
        if message_log:
            for message_list in message_log:
                if len(message_list) == 2:
                    self.string(message_list[0], message_list[1])
                else:
                    self.__internal(
                        f"Message List is not in the correct format: {message_list}"
                    )

    def function(self, func: callable):
        """
        A decorator that logs the execution details of a function, tracking its performance and providing runtime insights.
        
        Parameters:
            func (callable): The function to be decorated and monitored.
        
        Returns:
            callable: A wrapper function that logs execution metrics.
        
        Raises:
            TypeError: If the provided function is not callable.
        
        Example:
            @log.function
            def example_function():
                # Function implementation
                pass
        """
        if not callable(func):
            self.exception(f"Function {func.__name__} is not callable.", TypeError)

        def wrapper(*args, **kwargs):
            """
            Wrapper function that logs the execution of the decorated function.
            
            Tracks and logs the start, execution, and completion of a function with performance timing.
            
            Parameters:
                *args (tuple): Positional arguments passed to the decorated function.
                **kwargs (dict): Keyword arguments passed to the decorated function.
            
            Returns:
                Any: The original result of the decorated function.
            
            Raises:
                TypeError: If the decorated function is not callable.
            
            Notes:
                - Logs debug messages before and after function execution
                - Measures and logs the total execution time with microsecond precision
                - Preserves the original function's return value
            """
            start_time = time.perf_counter()
            func_args = ", ".join([str(arg) for arg in args] +
                                  [f"{k}={v}" for k, v in kwargs.items()])
            self.debug(f"Running the function {func.__name__}({func_args}).")
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            self.debug(f"{func.__name__}({func_args}) executed in {elapsed_time} -> returned {type(result).__name__}")
            return result

        return wrapper


log = Log({"log_level": DEBUG})
