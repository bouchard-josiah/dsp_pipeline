import numpy as np

from dsp_demo.FIR_Filter import Filters

class channelize:

    def __init__(
        self,

        spectrogram: np.ndarray,

        detect_mask: np.ndarray,

        FFT_taps: int = 4096,

        FIR_taps: int = 257,

        sample_rate: int = 50e6
     

    ):
        
        self.spectrogram = spectrogram

        self.detect_mask = detect_mask

        self.FFT_taps: int = FFT_taps

        self.spec_len: int = spectrogram.shape(0)

        self.signals_detected: int = 0

        self.FIR_taps: int  = FIR_taps
        
        self.filtered_sig_f = np.zeros([self.spec_len, self.FFT_taps], dtype = complex)

        self.detection_adj_count: int = 8

        self.detection_adj_sense: int = 6

        self.detect_sig_gap_sense: int = 3

        self.detect_times = np.zeros(self.signals_detected, dtype = int) #in time bins

        self.detect_freqs = np.zeros(self.signals_detected, dtype = int) #in freq bins

        #self.detect_chains = np.zeros(self.signals_detected, dtype = int)

        self.detect_bws = np.zeros(self.signals_detected, dtype = int) #width measured in bins

        self.sample_rate: int = sample_rate

    def find_sig_bws(self):

        start_again = 0
        self.signals_detected = 0
        checking: bool = True
        for t in range(0,self.spec_len):
            checking = True
            for f in range(0,self.FFT_taps):
                if (np.sum(self.detect_mask[t,f+self.detection_adj_count]) > 255*self.detection_adj_sense) :
                    
                    if start_again <= ((t*self.FFT_taps) + f):
                        
                        self.detect_freqs[self.signals_detected] = f

                        self.detect_times[self.signals_detected] = t

                        for n in range(f,self.FFT_taps):
                            if(np.sum(self.detect_mask[t,n+self.detect_sig_gap_sense]) < 255) and checking and ((n+self.detect_sig_gap_sense) < self.FFT_taps):
                            
                                checking = False

                                self.detect_bws[self.signals_detected] = n - self.detect_freqs[self.signals_detected]

                                start_again = t*self.FFT_taps + n+ self.detect_sig_gap_sense
                            
                            elif((n+self.detect_sig_gap_sense) >= self.FFT_taps):

                                start_again = t*self.FFT_taps + f + 1

                        self.signals_detected += 1
                    


    def gen_output_sig(self, signal_choice: int = 0):
        
        hz_per_bin = self.sample_rate/self.FFT_taps

        freq_offset = (self.detect_freqs[signal_choice] + self.detect_bws[signal_choice]/2)*hz_per_bin

        filter_bw = self.detect_bws[signal_choice]*hz_per_bin*1.10

        Filters_loc = Filters(spectrogram=self.spectrogram, FFT_taps=self.FFT_taps, FIR_taps=self.FIR_taps, sample_rate=self.sample_rate, freq_shift=freq_offset, filter_bw=filter_bw)

        Filters.freq_shift()

        Filters.gen_windowed_FIR_f()

        Filters.overlap_add()

        for t in range(0,self.spec_len):
            self.filtered_sig_f[t,:] = np.fft.fft(Filters.filtered_signal_t[t:t+self.FFT_taps])
