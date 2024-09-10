# To use this, make a copy, the name it TEST_{FILENAME} where {FILENAME} is replaced

import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


unittest.main()
