import time


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.time = 0

    def rerun(self):
        self.time = 0
        self.start_time = time.time()

    def stop(self):
        self.time = time.time() - self.start_time

    def get(self):
        return self.time
