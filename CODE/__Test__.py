import unittest
import sys  #  MUST-USE FOR ALL CODE TO WORK WITH EXE


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == "__main__":
    unittest.main()
