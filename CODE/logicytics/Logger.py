from __future__ import annotations

import inspect
import logging
import os
import time
from datetime import datetime
from typing import Type

import colorlog


class Log:
    """
    A logging class that supports colored output using the colorlog library.
    """

    def __init__(self, config: dict = None):
        """
        Initializes the Log class with the given configuration.

        :param config: A dictionary containing configuration options.
        """
        config = config or {
            "filename": "../ACCESS/LOGS/Logicytics.log",
            "use_colorlog": True,
            "log_level": "INFO",
            "debug_color": "cyan",
            "info_color": "green",
            "warning_color": "yellow",
            "error_color": "red",
            "critical_color": "red",
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
        self.filename = config.get("filename", "../ACCESS/LOGS/Logicytics.log")
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
                "CRITICAL": config.get("critical_color", "red"),
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
            self.raw(
                "|     Timestamp     |  LOG Level  |"
                + " " * 71
                + "LOG Messages"
                + " " * 71
                + "|"
            )
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
        Logs an internal message. Internal messages are displayed in the console only.

        :param message: The internal message to be logged.
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

    def raw(self, message):
        """
        Logs a raw message directly to the log file.
        This should only be called from within the Log class.
        So do not use this method in your code.

        :param message: The raw message to be logged.
        """
        frame = inspect.stack()[1]
        if frame.function == "<module>":
            self.__internal(
                f"Raw message called from a non-function - This is not recommended"
            )
        if message != "None" and message is not None:
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
        Logs a newline separator in the log file.
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
        self.raw(
            f"[{self.__timestamp()}] > INFO:     | {self.__trunc_message(str(message))}"
        )

    def warning(self, message):
        """
        Logs a warning message.

        :param message: The warning message to be logged.
        """
        if self.color and message != "None" and message is not None:
            colorlog.warning(str(message))
        self.raw(
            f"[{self.__timestamp()}] > WARNING:  | {self.__trunc_message(str(message))}"
        )

    def error(self, message):
        """
        Logs an error message.

        :param message: The error message to be logged.
        """
        if self.color and message != "None" and message is not None:
            colorlog.error(str(message))
        self.raw(
            f"[{self.__timestamp()}] > ERROR:    | {self.__trunc_message(str(message))}"
        )

    def critical(self, message):
        """
        Logs a critical message.

        :param message: The critical message to be logged.
        """
        if self.color and message != "None" and message is not None:
            colorlog.critical(str(message))
        self.raw(
            f"[{self.__timestamp()}] > CRITICAL: | {self.__trunc_message(str(message))}"
        )

    def string(self, message, type: str):
        """
        Logs a message with a specified type. Supported types are 'debug', 'info', 'warning', 'error', 'critical'
        as well as the aliases 'err', 'warn', and 'crit'.

        :param message: The message to be logged.
        :param type: The type of the log message.
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
        Logs an exception message.

        This is not recommended for use in production code, as it raises the exception after logging it.
        Use Log().error() instead.

        :param message: The exception message to be logged.
        :param exception_type: The type of exception to raise.
        """
        if self.color and message != "None" and message is not None:
            self.raw(
                f"[{self.__timestamp()}] > EXCEPTION:| {self.__trunc_message(f'{message} -> Exception provoked: {str(exception_type)}')}"
            )
        raise exception_type(message)

    def parse_execution(self, message_log: list[list[str, str]]):
        """
        Parses and logs a list of messages with their corresponding log types.
        Only use this method if you have a list of lists where each inner list contains a message and its log type.
        Use case include "Execute.script()" function.

        :param message_log: A list of lists, where each inner list contains a message and its log type.
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
        A decorator that logs the execution of a function,
        including its start time, end time, and elapsed time.

        :param func: The function to be decorated.
        :return: The wrapper function.
        """
        if not callable(func):
            self.exception(f"Function {func.__name__} is not callable.", TypeError)

        def wrapper(*args, **kwargs):
            """
            Wrapper function that logs the execution of the decorated function.

            :param args: Positional arguments for the decorated function.
            :param kwargs: Keyword arguments for the decorated function.
            :raises TypeError: If the provided function is not callable.
            :return: The result of the decorated function.
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
