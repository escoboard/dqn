import time


def get_timestamp(milliseconds=False):
    if milliseconds:
        return time.time()
    else:
        return int(time.time())