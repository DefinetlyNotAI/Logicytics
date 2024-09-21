import unittest
from CODE.wifi_stealer import get_password


class WiFi(unittest.TestCase):
    def test_get_password(self):
        print("Running test...")
        result = get_password("GEMS-WGP")
        print(f"Result: {result}")
        self.assertIsNotNone(result)
        print("Test completed.")


if __name__ == "__main__":
    unittest.main()
