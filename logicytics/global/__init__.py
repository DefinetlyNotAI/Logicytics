"""Globally available, host-bound infrastructure loaded only by explicit consumers.

The directory name follows the product architecture.  Because ``global`` is a
Python keyword, consumers load submodules through the public root exports or
``importlib.import_module("logicytics.global.<name>")``.
"""
