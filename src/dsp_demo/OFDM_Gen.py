'''
OFDM signal generation


This module contains class objects to generate an OFDM signal. 
It uses BPSK modulation and has functions to generate a real, hermitian symmetric, signal
and an analytic signal whose spectrogram does not show negative frequency components (SDR-like). 
There is functionality for both FFT-style modulation and discrete-time modulation for modulation of an analytic signal 
Other functions allow for spectrogram generation, in window or saved as a PNG, and functionality for adding AWGN  


Typical usage:

    cfg = configs()

    OFDM_loc = OFDM(cfg)

    OFDM_loc.msg_gen()

    OFDM_loc.IFFT_sig_BPSK_mod(cfg)

    OFDM_loc.add_noise(cfg)

    OFDM_loc.gen_spectro(cfg)

    OFDM_loc.print_spectro(cfg,spec_location=OFDM_loc.spec_location)


Dependencies:
    -numpy 

    -dataclasses.dataclass

    -random (for message generation and AWGN)

    -string (for plotting)

    -pathlib.Path (for plotting)

    -matplotlib (for plotting)

    -python3-tk (in ubuntu for plotting spectrograms in window) 


'''
import numpy as np

from dataclasses import dataclass

import random

#import string

from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt





@dataclass
class configs:

    #signal_bw: float = 10e3 #Generated signal bandwidth

    FFT_taps: int = 4096 #Spectrogram frequency fidelity and sets new sample rate in IFFT modulation because of static subcarrier spacing

    spec_len: int = 1000 #Spectrogram length in time bins

    scs: int = 30e3 #Sub carrier spacing of the OFDM carriers

    OFDM_symb_len: float = 1/scs #Symbol length in seconds which is directly related to subcarrier spacing

    sample_rate: int = 100e6 #Subject to be reconfigured when using FFT based modulation to avoid resampling

    noise_pwr: float = .9 #Noise floor (Also arbitrary)





class OFDM:

    def __init__(
            
        self, 
            
        cfg: configs,

        msg_carriers: int = 750,

        tx_symbs: int = 400,

        f_offset: float = 25e6,

        t_offset_bins: float =  0, #Time bins from receive start to start of generated signal

        save_spec: bool = True,

        spec_location: str = str(Path.home()) + "/dsp_demo/results/Unfiltered_spectrogram.png",

        sig_gain: float = 400 #Signal power (arbitrary linear unit)
    
    ):


        self.msg_carriers: int = msg_carriers #This is how many subcarriers our message will take up per OFDM symbol. This is what sets the occupied signal bw. Messages longer than this will be transmitted in proceeding symbols.

        self.tx_symbs: int = tx_symbs  #This is the amount of OFDM symbols the transmitted signal occupies in the "collected" signal

        self.f_offset: float = f_offset  #-15e6#25e3 #center frequency offset from baseband to center of tx bw

        self.t_offset: float =  (cfg.FFT_taps*t_offset_bins)/cfg.sample_rate#t_offset #((cfg.spec_len/4)*cfg.FFT_taps)/cfg.sample_rate  #seconds. This is when the transmission of the occupied symbols begins in seconds

        self.sig_gain: float = sig_gain #Signal power (arbitrary linear unit)
        
        self.rx_sig_len = cfg.FFT_taps * cfg.spec_len #received signal length in samples
        
        self.signal_t = np.zeros(self.rx_sig_len, dtype=complex) #This stores the modulated signal in time

        self.msg_len = self.msg_carriers*self.tx_symbs#int(cfg.signal_bw/cfg.scs) #This is the length of the message that is being modulated. The same message is repeated in every OFDM symbol for the length of the spectrogram in time. Each subcarrier has 1 bit in BPSK

        self.msg_bin = np.zeros(self.msg_len, dtype=bool) #This stores the binary message that will be modulated

        self.sample_rate = cfg.sample_rate

        self.spectrogram = np.zeros([cfg.spec_len, cfg.FFT_taps],dtype = complex) #initalizing our matrix container that will be plotted as spectrogram (time vs. freq)

        self.save_spec: bool = save_spec

        self.spec_location: str = spec_location

        if self.t_offset > (self.rx_sig_len/self.sample_rate): #covering for bad config
            self.t_offset = (self.rx_sig_len/2)/self.sample_rate
        else:
            self.t_offset = self.t_offset


    def msg_gen(self):

        for x in range(0,self.msg_len):
            self.msg_bin[x] = random.choice([True, False]) #populating our transmission symbols
    

    def real_sig_BPSK_mod(self, cfg: configs): #This function modulates our message and generates a real sampled signal with hermitian symmetry


        for symb in range(0,self.tx_symbs):

            
            symb_start_samp = (self.sample_rate+ symb*cfg.OFDM_symb_len)*cfg.sample_rate

            symb_end_samp = (self.sample_rate+ (symb+1)*cfg.OFDM_symb_len)*cfg.sample_rate


            for t in range(symb_start_samp, symb_end_samp):

                if t%cfg.FFT_taps == 0:

                    print(t/cfg.FFT_taps) 

                for n in range(0,self.msg_carriers):

                    f = ((cfg.scs*n) + self.f_offset - (self.msg_len*cfg.scs/2)) #f is our current subcarrier frequency

                    if self.msg_bin[((self.msg_carriers*symb)+n)%self.msg_len] == True: #The iteration of msg_bin accounts for mismatched carrier amount and message length if configured improperly.
                    
                        self.signal_t.real[t] += self.sig_gain*np.cos(2*np.pi*f*t/cfg.sample_rate) #BPSK sample generation with just cosine

                    else:
                    
                        self.signal_t.real[t] += (-1)*self.sig_gain*np.cos(2*np.pi*f*t/cfg.sample_rate) #When false, generate a signal with 180 degree phase shift from our True bit signal
  
    def analytic_sig_BPSK_mod(self, cfg: configs): #This function modulates a signal that is seen as being analytic. This means the signal does not have hermitian symmetry. It uses exponentials using numpy instead of sine and cosine
  
        for symb in range(0,self.tx_symbs):
            
            symb_start_samp = (self.sample_rate+ symb*cfg.OFDM_symb_len)*cfg.sample_rate

            symb_end_samp = (self.sample_rate+ (symb+1)*cfg.OFDM_symb_len)*cfg.sample_rate

            for t in range(symb_start_samp, symb_end_samp):

                if t%cfg.FFT_taps == 0:

                    print(t/cfg.FFT_taps) 

                for n in range(0,self.msg_carriers):

                    f = ((cfg.scs*n) + self.f_offset - (self.msg_len*cfg.scs/2)) #f is our current subcarrier frequency

                    expo = 1j*2*np.pi*f*t/cfg.sample_rate

                    if self.msg_bin[((self.msg_carriers*symb)+n)%self.msg_len] == True: #The iteration of msg_bin accounts for mismatched carrier amount and message length if configured improperly.
                    
                        self.signal_t[t] += np.exp(expo)*self.sig_gain

                    else:
                    
                        self.signal_t[t] += (-1)*np.exp(expo)*self.sig_gain


    def IFFT_sig_BPSK_mod(self, cfg: configs): #Populates carrier symbols in the frequency domain and then takes the IFFT. Exact sample rate required for this so that no resampling is needed and carrier symbols are relegated to EXACTLY 1 FFT matrix element. This is more alike how current OFDM radios modulate signals

        small_samp_rate = int(cfg.scs*cfg.FFT_taps) #This is what the sample rate for the individual small tx signal would need to be for there to be exactly 1 FFT tap per subcarrier.

        if self.sample_rate > small_samp_rate:
            sig_bw_part = int((self.sample_rate/small_samp_rate)+1) #how many small sample rates are in the original configured sample rate
        else:
            sig_bw_part = 1

        self.sample_rate = small_samp_rate*sig_bw_part #Multiplying the small sample rate to be close to the original configured sample rate whilst keeping our carriers per FFT tap an integer (or very close)
        
        #if self.t_offset > ((self.rx_sig_len)/self.sample_rate): #covering for bad config with new sample rate
            #self.t_offset = (self.rx_sig_len/2)/self.sample_rate

        symbol_dur_samps = self.sample_rate*cfg.OFDM_symb_len

        symbol_dur_bins = int(symbol_dur_samps/cfg.FFT_taps) #This should be exact from changing the samp rate anyway. Int for any weird rounding from weird configs

        hz_per_bin = (self.sample_rate*1.0)/cfg.FFT_taps

        center_bin_f = int(self.f_offset/hz_per_bin)

        start_bin_f = int((cfg.FFT_taps/2) + int(center_bin_f - (self.msg_carriers/2)))

        end_bin_f = int(start_bin_f + self.msg_carriers*sig_bw_part)

        #t_offset_symb = int((self.t_offset*self.sample_rate)/(cfg.FFT_taps)) #This will align the configured offset, in time, to be a delay, from start of collection, equal to the start of the next PRB -> physical resource block, or in this case, time bin

        start_bin_t = int((self.t_offset*self.sample_rate)/cfg.FFT_taps)



        #self.signal_t[0] = complex(1,1)

        for symb in range(0,self.tx_symbs):
            
            FFT_sig = np.zeros(cfg.FFT_taps, dtype = complex)

            IFFT_sig_hold = np.zeros(cfg.FFT_taps, dtype = complex)
            
            for t_bin in range(0,symbol_dur_bins):

                for f_bin in range(start_bin_f, end_bin_f):
            
                    if self.msg_bin[((self.msg_carriers*symb)+f_bin)%self.msg_len] == True:
                        FFT_sig[f_bin] = complex(self.sig_gain, self.sig_gain)
                    else:
                        FFT_sig[f_bin] = complex((-1*self.sig_gain),(-1*self.sig_gain))

                FFT_sig = np.fft.ifftshift(FFT_sig)
               
                IFFT_sig_hold = np.fft.ifft(FFT_sig)

                start_time_samp = (start_bin_t + (symb*symbol_dur_bins)+t_bin)*cfg.FFT_taps #(t_offset_symb*symbol_dur_bins*cfg.FFT_taps)+(t_bin

                self.signal_t[start_time_samp:start_time_samp + cfg.FFT_taps] = IFFT_sig_hold

  

    def add_noise(self, cfg: configs, signal: np.ndarray | None = None): #adds AWGN to give our signal an imperfect SNR

        if signal == None:

            signal = self.signal_t

        for x in range(0, len(signal)):

            real_noise = np.random.normal(0, np.sqrt(cfg.noise_pwr)/2)
            self.signal_t.real[x] = signal.real[x] + real_noise

            imag_noise = np.random.normal(0, np.sqrt(cfg.noise_pwr)/2)
            self.signal_t.imag[x] = signal.imag[x] + imag_noise


    def gen_spectro(self, cfg: configs, signal: np.ndarray | None = None):
        
        if signal == None:

            signal = self.signal_t

            spec_len = int(len(signal)/cfg.FFT_taps)

        else:

            spec_len = cfg.spec_len

        for x in range(0,spec_len):

            bin = np.fft.fft(signal[ (x*cfg.FFT_taps) : (((x+1)*cfg.FFT_taps))]) #DFT taken over a set of samples with the length of our FFT Taps
            #bin = np.fft.fftshift(bin)
            self.spectrogram[x,0:cfg.FFT_taps] = bin #np.abs(bin)#(np.fft.fftshift(bin))) #np.abs(bin) #magnitude of our complex frequency vector
            #or y in range(0,cfg.FFT_taps):



    def print_spectro(self, cfg: configs, spec_location: str | None = None, spectrogram: np.ndarray | None = None):

        if spectrogram == None:

            spectrogram = self.spectrogram

        if spec_location == None:

            spec_location = self.spec_location
        
        plt.figure(figsize=(20,12)) #figure size
        printer = np.zeros([spectrogram.shape[0],spectrogram.shape[1]], dtype = complex)

        for n in range(0,spectrogram.shape[0]):
            printer[n,:] = np.fft.fftshift(spectrogram[n,:])
        
        plt.imshow(np.abs(printer),cmap='viridis') #setting our colormap
        
        if self.save_spec == True: #Can save the plot as a png or shown in a window. Configured both path and choice in configs
            
            plt.savefig(spec_location, dpi=300)
        
        else:
            
            plt.show()







    


