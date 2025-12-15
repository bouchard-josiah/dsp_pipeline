import numpy as np


class Filters:

    def __init__(   
        
        self,

        spectrogram: np.ndarray,

        FFT_taps: int = 4096,

        freq_shift: float = 0, #hz

        FIR_taps: int = 257,

        sample_rate: int = 50e6,

        filter_bw: float = 10e6

    ):

        self.spectrogram = spectrogram

        self.FFT_taps = FFT_taps

        self.FIR_taps: int = FIR_taps

        self.freq_shift: float = freq_shift #hz

        self.filtered_signal_t = np.zeros(spectrogram.size()+FIR_taps - 1, dtype = complex)

        self.FIR_filter_t = np.zeros(FIR_taps,dtype = float)

        self.FIR_filter_f = np.zeros(FIR_taps,dtype = float)

        self.sample_rate: int = sample_rate

        self.filter_bw: float = filter_bw

        self.shifted_sig_t = np.zeros(self.spectrogram.size(),dtype = complex)

        self.shifted_sig_f = np.zeros([self.spectrogram.shape(0), self.spectrogram.shape(1)], dtype = complex)


    def freq_shift(self):
        for t in range(0,self.spectrogram.shape(0)):
            self.shifted_sig_t[t:t+self.FFT_taps] = np.fft.ifft(self.spectrogram[t,:])
        
        for n in range(0,self.shifted_sig_t.size()):
            expo = (((-1j)*2*np.pi*self.freq_shift*n)/self.sample_rate)
            self.shifted_sig_t = self.shifted_sig_t*(np.exp(expo))

        for t in range(0,self.spectrogram.shape(0)):
            self.shifted_sig_f[t,:] = np.fft.fft(self.shifted_sig_t[t:t+self.FFT_taps]) 


    def gen_windowed_FIR_f(self):

        B = .5/(self.sample_rate/self.filter_bw)

        for n in range(0,self.FIR_taps):
            self.FIR_filter_t[n] = B*np.sinc(B*(n-((self.FIR_taps-1)/2)))

        self.FIR_filter_t = self.FIR_filter_t * np.hanning(self.FIR_taps)
        
        self.FIR_filter_t /= np.sum(self.FIR_filter)

        self.FIR_filter_f = np.fft.fft(self.FIR_filter_t)


    def overlap_add(self,unfiltered_sig_t: np.ndarray):

        overlap = np.zeros(self.FIR_taps-1, dtype = complex)

        padded_filter_f =np.zeros(self.FFT_taps, dtype = float)

        padded_filter_f[0:self.FIR_taps] = self.FIR_filter_f 
        
        hold = np.zeros(self.FFT_taps, dtype = complex)
        for n in range(0,int(unfiltered_sig_t.size()/self.FFT_taps)):

            hold = np.fft.ifft(complex(np.fft.fft(unfiltered_sig_t[n:n+self.FFT_taps].real)*padded_filter_f, np.fft.fft(unfiltered_sig_t[n:n+self.FFT_taps].imag)*padded_filter_f))
            
            self.filtered_signal_t[(n*self.FFT_taps):(n+1)*self.FFT_taps] = hold

            if (n < (int(unfiltered_sig_t.size()/self.FFT_taps) - 1) and (n > 0)):

                self.filtered_signal_t[(n*self.FFT_taps)-(self.FIR_taps-1):n*self.FFT_taps] += overlap

            overlap = hold[self.FFT_taps-(self.FIR_taps-1):self.FFT_taps]

            if n == (int(unfiltered_sig_t.size()/self.FFT_taps) - 1):
                self.filtered_signal_t[self.filtered_signal_t.size()-(self.FIR_taps-1):self.filtered_signal_t.size()] = overlap

                       

        