def is_leap_year(year):
    """if (year is not divisible by 4) then (it is a common year)
    else if (year is not divisible by 100) then (it is a leap year)
    else if (year is not divisible by 400) then (it is a common year)
    else (it is a leap year)"""
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    elif year % 400 != 0:
        return False
    else:
        return True

def is_valid_date(day,month,year):
    if (year<=0):
        return False
    if (month==1 or month==3 or month==5 or month==7 or month==8 or month==10 or month==12):
        if day<=31 and day>0:
            return True
        else:
            return False
    elif (month==4 or month==6 or month==9 or month==11):
        if day<=30 and day>0:
            return True
        else:
            return False
    elif (month==2):
        if not is_leap_year(year):
            if day<=28 and day>0:
                return True
            else:
                return False
        else:
            if day<=29 and day>0:
                return True
            else:
                return False
    return False

print(is_valid_date(31,4,2016))
print(is_valid_date(0,5,2004))
print(is_valid_date(31,12,1999))
print(is_valid_date(29,2,2004))
print(is_valid_date(29,2,2005))
print(is_valid_date(29,2,2000))
print(is_valid_date(29,2,1900))
print(is_valid_date(1,1,0))
print(is_valid_date(1,13,2017))
print(is_valid_date(4,0,2016))
