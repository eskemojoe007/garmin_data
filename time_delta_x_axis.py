import datetime

def strfdelta(td,fmt):
    seconds = int(td.total_seconds())
    d = {'total_seconds':seconds}
    d['total_minutes'],d['seconds'] = divmod(seconds,60)
    d['hours'],rem = divmod(seconds,3600)
    d['minutes'],d['seconds'] = divmod(rem,60)

    d['total_seconds'] = '{:02d}'.format(d['total_seconds'])
    d['total_minutes'] = '{:02d}'.format(d['total_minutes'])
    d['hours'] = '{:02d}'.format(d['hours'])
    d['minutes'] = '{:02d}'.format(d['minutes'])
    d['seconds'] = '{:02d}'.format(d['seconds'])

    return fmt.format(**d)

def td_formatter(x,pos,fmt='{total_minutes}:{seconds}'):
    return strfdelta(x,fmt)
