import numpy as np


class Filters:

    def __init__(   
        
        self,

        spectrogram: np.ndarray,

        FFT_taps: int = 4096,

        freq_offset: float = 0, #hz

        FIR_taps: int = 257,

        sample_rate: int = 50e6,

        filter_bw: float = 10e6

    ):

        self.spectrogram = spectrogram

        self.FFT_taps = FFT_taps

        self.FIR_taps: int = FIR_taps

        self.freq_offset: float = freq_offset #hz

        self.filtered_signal_t = np.zeros((self.spectrogram.size+FIR_taps - 1), dtype = complex)

        self.FIR_filter_t: np.ndarray

        self.FIR_filter_f: np.ndarray 

        self.sample_rate: int = sample_rate

        self.filter_bw: float = filter_bw

        self.shifted_sig_t = np.zeros(self.spectrogram.size,dtype = complex)

        self.shifted_sig_f = np.zeros([self.spectrogram.shape[0], self.spectrogram.shape[1]], dtype = complex)

        self.filtered_sig_f = np.zeros([self.spectrogram.shape[0], self.spectrogram.shape[1]], dtype = complex)

        self.filter_FFT_taps: int = 4096



    def freq_shift(self): #input is a spectrogram in the freq domain
        for t in range(0,self.spectrogram.shape[0]):
            self.shifted_sig_t[t*self.FFT_taps:(t+1)*self.FFT_taps] = np.fft.ifft(self.spectrogram[t,:])
        
        for n in range(0,self.shifted_sig_t.size):
            expo = (((-1j)*2*np.pi*self.freq_offset*n)/self.sample_rate)
            self.shifted_sig_t[n] = self.shifted_sig_t[n]*(np.exp(expo))

        for t in range(0,self.spectrogram.shape[0]):
            self.shifted_sig_f[t,:] = np.fft.fft(self.shifted_sig_t[t*self.FFT_taps:(t+1)*self.FFT_taps]) 


    def gen_windowed_FIR_f(self):

        B = .5/(self.sample_rate/self.filter_bw)

        self.FIR_filter_t= np.zeros(self.filter_FFT_taps,dtype = float)

        self.FIR_filter_f = np.zeros(self.filter_FFT_taps,dtype = complex)

        for n in range(0,self.FIR_taps):
            self.FIR_filter_t[n] = B*np.sinc(B*(n-((self.FIR_taps-1)/2)))

        self.FIR_filter_t = self.FIR_filter_t * np.hanning(self.FIR_filter_t.size)
        
        self.FIR_filter_t[:] /= np.sum(self.FIR_filter_t)

        self.FIR_filter_f = np.fft.fft(self.FIR_filter_t)


    def find_fft_taps(self):

        n: int = 0

        while(self.FFT_taps > (2**n)):
            n += 1
                
        self.filter_FFT_taps = 2**n
                
    
    def overlap_add_filter(self):

        self.freq_shift()

        self.find_fft_taps()

        self.gen_windowed_FIR_f()
        
        overlap = np.zeros(self.FIR_taps-1, dtype = complex)

        hold = np.zeros(self.filter_FFT_taps, dtype = complex)

        for n in range(0,int(self.shifted_sig_t.size/self.filter_FFT_taps)):

            hold[:] = np.fft.ifft(np.fft.fft(self.shifted_sig_t[n*self.filter_FFT_taps:(n+1)*self.filter_FFT_taps])*self.FIR_filter_f)

            self.filtered_signal_t[(n*self.filter_FFT_taps):(n+1)*self.filter_FFT_taps] = hold

            if ((n > 0)):

                self.filtered_signal_t[(n*self.filter_FFT_taps) - (self.FIR_taps - 1) : (n*self.filter_FFT_taps)] += overlap

            overlap = self.filtered_signal_t[(n*self.filter_FFT_taps) : (n*self.filter_FFT_taps) + (self.FIR_taps - 1)]

        for n in range(0,self.filtered_sig_f.shape[0]):

            self.filtered_sig_f[n,:] = np.fft.fft(self.filtered_signal_t[n*self.FFT_taps:(n+1)*self.FFT_taps])

            



                       

        