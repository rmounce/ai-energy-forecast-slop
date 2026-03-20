import pandas as pd
import pytz
local_tz = pytz.timezone('Australia/Adelaide')
idx = pd.date_range('2025-01-01', '2025-12-31', freq='30min', tz='UTC').tz_convert(local_tz)

v = idx.map(lambda x: x.dst() > pd.Timedelta(0)).astype(int)
print(sum(v))
