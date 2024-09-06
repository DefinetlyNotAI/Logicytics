import unittest
from unittest import mock

import Logicytics


class TestLogicytics(unittest.TestCase):
    def setUp(self):
        self.checker = Logicytics.Check()

    @mock.patch('ctypes.windll.shell32.IsUserAnAdmin')
    def test_admin(self, mock_admin):
        # Test when user is an admin
        mock_admin.return_value = True
        self.assertTrue(self.checker.admin())

        # Test when user is not an admin
        mock_admin.return_value = False
        self.assertFalse(self.checker.admin())

        # Test when ctypes raises AttributeError
        mock_admin.side_effect = AttributeError
        self.assertFalse(self.checker.admin())

    @mock.patch.object(Logicytics.Actions, 'run_command')
    def test_uac(self, mock_run_command):
        # Test when UAC is enabled
        mock_run_command.return_value = "1\n"
        self.assertTrue(self.checker.uac())

        # Test when UAC is disabled
        mock_run_command.return_value = "0\n"
        self.assertFalse(self.checker.uac())

        # Test unexpected return value
        mock_run_command.return_value = "unexpected\n"
        with self.assertRaises(ValueError):
            self.checker.uac()


if __name__ == '__main__':
    unittest.main()
