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

    signal_bw: float = 10e3 #Generated signal bandwidth

    f_offset: float = 15e3#25e3 #center frequency offset from baseband to center of tx bw

    scs: int = 30 #Sub carrier spacing of the OFDM carriers

    sample_rate: int = 50e3 

    FFT_taps: int = 1024 #Spectrogram frequency fidelity

    spec_len: int = 250 #Spectrogram length in time bins

    sig_gain: float = 40 #Signal power (arbitrary linear unit)

    noise_gain: float = .9 #Noise floor (Also arbitrary)

    save_spec: bool = True

    spec_location: string = str(Path.home()) + "/dsp_demo/results/spectrogram4.png"



class OFDM:

    def __init__(self, config: configs):

        self.sig_len = config.FFT_taps * config.spec_len
        
        self.signal_t = np.zeros(self.sig_len, dtype=complex) #This stores the modulated signal in time

        self.msg_len = int(config.signal_bw/config.scs) #This is the length of the message that is being modulated. The same message is repeated in every OFDM symbol for the length of the spectrogram in time. Each subcarrier has 1 bit in BPSK

        self.msg_bin = np.zeros(self.msg_len, dtype=bool) #This stores the binary message that will be modulated


    def msg_gen(self):

        for x in range(0,self.msg_len):
            self.msg_bin[x] = random.choice([True, False]) #populating our transmission symbols
    

    def real_sig_BPSK_mod(self, config: configs): #This function modulates our message and generates a real sampled signal with hermitian symmetry

        for t in range(0,self.sig_len):
            
            if t%config.FFT_taps == 0:

                print(t/config.FFT_taps) 

            for n in range(0,self.msg_len):

                f = ((config.scs*n) + config.f_offset - (self.msg_len*config.scs/2)) #f is our current subcarrier frequency

                if self.msg_bin[n]== True:
                
                    self.signal_t.real[t] += config.sig_gain*np.cos(2*np.pi*f*t/config.sample_rate) #BPSK sample generation with just cosine

                else:
            
                    self.signal_t.real[t] += (-1)*config.sig_gain*np.cos(2*np.pi*f*t/config.sample_rate) #When false, generate a signal with 180 degree phase shift from our True bit signal

    def analytic_sig_BPSK_mod(self, config: configs): #This function modulates a signal that is seen as being analytic. This means the signal does not have hermitian symmetry. It uses exponentials using numpy instead of sine and cosine
        for t in range(0,self.sig_len):
            
            if t%config.FFT_taps == 0:

                print(t/config.FFT_taps)

            for n in range(0,self.msg_len):

                f = ((config.scs*n) + config.f_offset - (self.msg_len*config.scs/2)) #current subcarrier frequency

                expo = 1j*2*np.pi*f*t/config.sample_rate #calculates our exponential for the current sample and frequency

                if self.msg_bin[n]== True:
                
                    self.signal_t[t] += np.exp(expo)*config.sig_gain

                else:
            
                    self.signal_t[t] += (-1)*np.exp(expo)*config.sig_gain #Again, phase shifted 180 degrees from the true signal
  

    def add_noise(self, config: configs): #adds AWGN to give our signal an imperfect SNR

        for x in range(0,self.sig_len):

            real_noise = np.random.normal(0, np.sqrt(config.noise_gain)/2)
            self.signal_t.real[x] = self.signal_t.real[x] + real_noise

            imag_noise = np.random.normal(0, np.sqrt(config.noise_gain)/2)
            self.signal_t.imag[x] = self.signal_t.imag[x] + imag_noise


    def gen_spectro(self, config: configs):

        spectrogram = np.zeros([config.spec_len, config.FFT_taps],dtype = float) #initalizing our matrix container that will be plotted as spectrogram (time vs. freq)

        for x in range(0,config.spec_len):

            bin = np.fft.fft(self.signal_t[(x*config.FFT_taps):(((x+1)*config.FFT_taps))]) #DFT taken over a set of samples with the length of our FFT Taps
            
            spectrogram[x,0:config.FFT_taps] = np.abs((np.fft.fftshift(bin))) #np.abs(bin) #magnitude of our complex frequency vector
            #or y in range(0,config.FFT_taps):

        plt.figure(figsize=(20,12)) #figure size
        
        plt.imshow(spectrogram,cmap='viridis') #setting our colormap
        
        if config.save_spec == True: #Can save the plot as a png or shown in a window. Configured both path and choice in configs
            
            plt.savefig(config.spec_location, dpi=300)
        
        else:
            
            plt.show()   







    


