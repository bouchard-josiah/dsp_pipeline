import numpy as np


class Filters:

    def __init__(   
        
        self,

        spectrogram: np.ndarray,

        FFT_taps: int = 4096,

        freq_shift: float = 0, #hz

        FIR_taps: int = 256,

        sample_rate: int = 50e6,

        filter_bw: float = 10e6

    ):

        self.FFT_taps = FFT_taps

        self.freq_shift: float = freq_shift #hz

        self.filtered_signal_t = np.zeros(spectrogram.size()+FIR_taps, dtype = complex)

        self.sample_rate: int = sample_rate

        self.filter_bw: float = filter_bw


    def freq_shift(self):


    def gen_windowed_FIR_f(self):

        B = .5/(self.sample_rate/self.filter_bw)



    def overlap_add_f(self):

