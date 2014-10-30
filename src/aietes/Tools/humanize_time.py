__author__ = 'bolster'
# !/usr/bin/env python

INTERVALS = [1, 60, 3600, 86400, 604800, 2419200, 29030400]
NAMES = [('second', 'seconds'),
         ('minute', 'minutes'),
         ('hour', 'hours'),
         ('day', 'days'),
         ('week', 'weeks'),
         ('month', 'months'),
         ('year', 'years')]


def humanize_time(amount, units=None):
    '''
       Divide `amount` in time periods.
       Useful for making time intervals more human readable.

       >>> humanize_time(173, "hours")
       [(1, 'week'), (5, 'hours')]
       >>> humanize_time(17313, "seconds")
       [(4, 'hours'), (48, 'minutes'), (33, 'seconds')]
       >>> humanize_time(90, "weeks")
       [(1, 'year'), (10, 'months'), (2, 'weeks')]
       >>> humanize_time(42, "months")
       [(3, 'years'), (6, 'months')]
       >>> humanize_time(500, "days")
       [(1, 'year'), (5, 'months'), (3, 'weeks'), (3, 'days')]
       >>> humanize_time(5.5, "seconds")
       [(5.5, 'seconds')]
       >>> humanize_time(5.0, "seconds")
       [(5.0, 'seconds')]
    '''
    result = []

    if units is None:
        units = 'seconds'

    unit = map(lambda a: a[1], NAMES).index(units)
    # Convert to seconds
    amount = amount * INTERVALS[unit]

    for i in range(len(NAMES) - 1, -1, -1):
        a = amount // INTERVALS[i]
        if a > 0:
            if amount - a < 1:
                result.append((amount, NAMES[i][1] ))
            else:
                result.append((int(a), NAMES[i][1 % int(a)]))
            amount -= (a) * INTERVALS[i]

    return result


def secondsToStr(s):
    humanified = humanize_time(s, "seconds")
    return ' '.join(["{} {}".format(tup[0], tup[1]) for tup in humanified])


if __name__ == "__main__":
    import doctest

    doctest.testmod()

