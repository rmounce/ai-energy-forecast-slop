import holidays
sa_holidays = holidays.AU(subdiv='SA')
import datetime
print(datetime.date(2026, 1, 1) in sa_holidays)
print(datetime.date(2026, 3, 11) in sa_holidays)
