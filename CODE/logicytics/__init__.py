from logicytics.Execute import *
from logicytics.Get import *
from logicytics.Logger import *
from logicytics.Checks import *
from logicytics.FileManagement import *
from logicytics.Flag import *


Execute = Execute()
Get = Get()
Check = Check()
FileManagement = FileManagement()
Flag = Flag()


DEBUG, VERSION, CURRENT_FILES, DELETE_LOGS = Get.config_data()
