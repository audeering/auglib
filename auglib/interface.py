import os
import glob
import audiofile as af


class OnlineTransform(object):
    def __init__(self, transform, sampling_rate : int):
        self.transform = transform
        self.sampling_rate = sampling_rate

    def __call__(self, signal):
        with AudioBuffer.from_array(signal, self.sampling_rate) as buf:
            self.transform(buf)
            transformed = buf.to_array()
        return transformed
