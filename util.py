import time

interval = 0.0


def chronometer(f):
    def measure_time():
        global interval
        initial_t = time.time()
        ret = f()
        interval = time.time() - initial_t
        return ret

    return measure_time


def print_chrono():
    global interval
    print("Elapsed_time %f" % interval)
    interval = 0.0
