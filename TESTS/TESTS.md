# Tests Directory

The `tests` directory contains all the test cases and testing-related files for the project.
This is where you should place unit tests, integration tests, and any other testing scripts
to ensure the quality and correctness of your code.
Here, we follow a specific naming convention to keep things organized and easily understandable.

## Naming Convention

- **Test Files:** Test files should be named with the prefix `TEST_` followed by a descriptive name related to the
  module or functionality they are testing. For example:
    - `TEST_login.py` for testing the login functionalities.
    - `TEST_database.py` for testing database interactions.

- **Test Classes and Methods:**
    - **Test Classes:** Should start with `Test_` followed by the name of the class or functionality being tested.
    - **Test Methods:** Should be named with the prefix `test_` followed by a descriptive name that indicates what the
      test is verifying. 
    - For example:
      ```python
      import unittest
      class TestLoginFunctionality(unittest.TestCase):
          def test_successful_login(self):
              ...
          
          def test_login_with_invalid_credentials(self):
              ...
      ```

Following these conventions helps in quickly identifying and understanding the purpose of each test and makes the
testing framework work more seamlessly with tools like pytest or unittest.
