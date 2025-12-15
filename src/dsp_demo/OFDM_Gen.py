'''
OFDM signal generation


This module contains class objects to generate an OFDM signal. 
It uses BPSK modulation and has functions to generate both a real, hermitian symmetric, signal
and an analytic signal whose spectrogram does not show negative frequency components (SDR-like).
Other functions allow for spectrogram generation, in window or saved as a PNG, and functionality for adding AWGN  


Typical usage:
    cfg = configs()

    OFDM = OFDM_BPSK(cfg)

    OFDM.msg_gen()

    real_sig_BPSK_mod(cfg)

    analytic_sig_BPSK_mod(cfg)

    OFDM.add_noise(cfg)

    OFDM.gen_spectro(cfg)


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

import string

from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt





@dataclass
class configs:

    #signal_bw: float = 10e3 #Generated signal bandwidth

    msg_carriers: int = 400 #This is how many subcarriers our message will take up per OFDM symbol. This is what sets the occupied signal bw. Messages longer than this will be transmitted in proceeding symbols.

    tx_symbs: int = 50 #This is the amount of OFDM symbols the transmitted signal occupies in the "collected" signal

    FFT_taps: int = 4096 #Spectrogram frequency fidelity and sets new sample rate in IFFT modulation because of static subcarrier spacing

    spec_len: int = 1000 #Spectrogram length in time bins

    scs: int = 30e3 #Sub carrier spacing of the OFDM carriers

    OFDM_symb_len: float = 1/scs #Symbol length in seconds which is directly related to subcarrier spacing

    sample_rate: int = 50e6 #Subject to be reconfigured when using FFT based modulation to avoid resampling

    f_offset: float = 0 #-15e6#25e3 #center frequency offset from baseband to center of tx bw

    t_offset: float =  ((spec_len/4)*FFT_taps)/sample_rate  #seconds. This is when the transmission of the occupied symbols begins in seconds

    sig_gain: float = 400 #Signal power (arbitrary linear unit)

    noise_gain: float = .9 #Noise floor (Also arbitrary)

    save_spec: bool = True

    spec_location: string = str(Path.home()) + "/dsp_demo/results/spectrogram5.png"



class OFDM:

    def __init__(self, config: configs):

        self.rx_sig_len = config.FFT_taps * config.spec_len #received signal length in samples
        
        self.signal_t = np.zeros(self.rx_sig_len, dtype=complex) #This stores the modulated signal in time

        self.msg_len = config.msg_carriers*config.tx_symbs#int(config.signal_bw/config.scs) #This is the length of the message that is being modulated. The same message is repeated in every OFDM symbol for the length of the spectrogram in time. Each subcarrier has 1 bit in BPSK

        self.msg_bin = np.zeros(self.msg_len, dtype=bool) #This stores the binary message that will be modulated

        self.sample_rate = config.sample_rate

        self.spectrogram = np.zeros([config.spec_len, config.FFT_taps],dtype = complex) #initalizing our matrix container that will be plotted as spectrogram (time vs. freq)

        if config.t_offset > (self.rx_sig_len/self.sample_rate): #covering for bad config
            self.t_offset = (self.rx_sig_len/2)/self.sample_rate
        else:
            self.t_offset = config.t_offset


    def msg_gen(self):

        for x in range(0,self.msg_len):
            self.msg_bin[x] = random.choice([True, False]) #populating our transmission symbols
    

    def real_sig_BPSK_mod(self, config: configs): #This function modulates our message and generates a real sampled signal with hermitian symmetry


        for symb in range(0,config.tx_symbs):

            
            symb_start_samp = (config.t_offset + symb*config.OFDM_symb_len)*config.sample_rate

            symb_end_samp = (config.t_offset + (symb+1)*config.OFDM_symb_len)*config.sample_rate


            for t in range(symb_start_samp, symb_end_samp):

                if t%config.FFT_taps == 0:

                    print(t/config.FFT_taps) 

                for n in range(0,config.msg_carriers):

                    f = ((config.scs*n) + config.f_offset - (self.msg_len*config.scs/2)) #f is our current subcarrier frequency

                    if self.msg_bin[((config.msg_carriers*symb)+n)%self.msg_len] == True: #The iteration of msg_bin accounts for mismatched carrier amount and message length if configured improperly.
                    
                        self.signal_t.real[t] += config.sig_gain*np.cos(2*np.pi*f*t/config.sample_rate) #BPSK sample generation with just cosine

                    else:
                    
                        self.signal_t.real[t] += (-1)*config.sig_gain*np.cos(2*np.pi*f*t/config.sample_rate) #When false, generate a signal with 180 degree phase shift from our True bit signal
  
    def analytic_sig_BPSK_mod(self, config: configs): #This function modulates a signal that is seen as being analytic. This means the signal does not have hermitian symmetry. It uses exponentials using numpy instead of sine and cosine
  
        for symb in range(0,config.tx_symbs):
            
            symb_start_samp = (config.t_offset + symb*config.OFDM_symb_len)*config.sample_rate

            symb_end_samp = (config.t_offset + (symb+1)*config.OFDM_symb_len)*config.sample_rate

            for t in range(symb_start_samp, symb_end_samp):

                if t%config.FFT_taps == 0:

                    print(t/config.FFT_taps) 

                for n in range(0,config.msg_carriers):

                    f = ((config.scs*n) + config.f_offset - (self.msg_len*config.scs/2)) #f is our current subcarrier frequency

                    expo = 1j*2*np.pi*f*t/config.sample_rate

                    if self.msg_bin[((config.msg_carriers*symb)+n)%self.msg_len] == True: #The iteration of msg_bin accounts for mismatched carrier amount and message length if configured improperly.
                    
                        self.signal_t[t] += np.exp(expo)*config.sig_gain

                    else:
                    
                        self.signal_t[t] += (-1)*np.exp(expo)*config.sig_gain


    def IFFT_sig_BPSK_mod(self, config: configs): #Populates carrier symbols in the frequency domain and then takes the IFFT. Exact sample rate required for this so that no resampling is needed and carrier symbols are relegated to EXACTLY 1 FFT matrix element. This is more alike how current OFDM radios modulate signals

        small_samp_rate = config.scs*config.FFT_taps #This is what the sample rate for the individual small tx signal would need to be for there to be exactly 1 FFT tap per subcarrier.

        if self.sample_rate > small_samp_rate:
            sig_bw_part = int((self.sample_rate/small_samp_rate)+1) #how many small sample rates are in the original configured sample rate
        else:
            sig_bw_part = 1

        self.sample_rate = small_samp_rate*sig_bw_part #Multiplying the small sample rate to be close to the original configured sample rate whilst keeping our carriers per FFT tap an integer (or very close)
        
        if self.t_offset > ((self.rx_sig_len)/self.sample_rate): #covering for bad config with new sample rate
            self.t_offset = (self.rx_sig_len/2)/self.sample_rate

        symbol_dur_samps = self.sample_rate*config.OFDM_symb_len

        symbol_dur_bins = int(symbol_dur_samps/config.FFT_taps) #This should be exact from changing the samp rate anyway. Int for any weird rounding from weird configs

        hz_per_bin = (self.sample_rate*1.0)/config.FFT_taps

        center_bin_f = int(config.f_offset/hz_per_bin)

        start_bin_f = int((config.FFT_taps/2) + int(center_bin_f - (config.msg_carriers/2)))

        end_bin_f = int(start_bin_f + config.msg_carriers*sig_bw_part)

        #t_offset_symb = int((config.t_offset*self.sample_rate)/(config.FFT_taps)) #This will align the configured offset, in time, to be a delay, from start of collection, equal to the start of the next PRB -> physical resource block, or in this case, time bin

        start_bin_t = int((self.t_offset*self.sample_rate)/config.FFT_taps)

        FFT_sig = np.zeros(config.FFT_taps, dtype = complex)

        IFFT_sig_hold = np.zeros(config.FFT_taps, dtype = complex)

        #self.signal_t[0] = complex(1,1)

        for symb in range(0,config.tx_symbs):
            for t_bin in range(0,symbol_dur_bins):

                for f_bin in range(start_bin_f, end_bin_f):
            
                    if self.msg_bin[((config.msg_carriers*symb)+f_bin)%self.msg_len] == True:
                        FFT_sig[f_bin] = complex(config.sig_gain, 0)
                    else:
                        FFT_sig[f_bin] = complex((-1*config.sig_gain),0)

                #FFT_sig = np.fft.fftshift(FFT_sig)

                IFFT_sig_hold = np.fft.ifft(FFT_sig)

                start_time_samp = (start_bin_t + (symb*symbol_dur_bins)+t_bin)*config.FFT_taps #(t_offset_symb*symbol_dur_bins*config.FFT_taps)+(t_bin

                self.signal_t[start_time_samp:start_time_samp + config.FFT_taps] = IFFT_sig_hold

  

    def add_noise(self, config: configs): #adds AWGN to give our signal an imperfect SNR

        for x in range(0,self.rx_sig_len):

            real_noise = np.random.normal(0, np.sqrt(config.noise_gain)/2)
            self.signal_t.real[x] = self.signal_t.real[x] + real_noise

            imag_noise = np.random.normal(0, np.sqrt(config.noise_gain)/2)
            self.signal_t.imag[x] = self.signal_t.imag[x] + imag_noise


    def gen_spectro(self, config: configs):

        for x in range(0,config.spec_len):

            bin = np.fft.fft(self.signal_t[ (x*config.FFT_taps) : (((x+1)*config.FFT_taps))] ) #DFT taken over a set of samples with the length of our FFT Taps
            
            self.spectrogram[x,0:config.FFT_taps] = bin #np.abs(bin)#(np.fft.fftshift(bin))) #np.abs(bin) #magnitude of our complex frequency vector
            #or y in range(0,config.FFT_taps):



    def print_spectro(self, config: configs):
        
        plt.figure(figsize=(20,12)) #figure size
        
        plt.imshow(np.abs(self.spectrogram),cmap='viridis') #setting our colormap
        
        if config.save_spec == True: #Can save the plot as a png or shown in a window. Configured both path and choice in configs
            
            plt.savefig(config.spec_location, dpi=300)
        
        else:
            
            plt.show()







    


