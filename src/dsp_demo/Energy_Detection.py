import numpy as np

import string

from pathlib import Path

from PIL import Image



class Energy_Detect:

    def __init__(
        self,

        spectrogram: np.ndarray,
    
        FFT_taps: int = 4096,
    
        spec_len: int = 1000,

        detect_thresh_db: float = 12,

        spec_img_path: string = str(Path.home()) + "/dsp_demo/results/spectrogram4.png",

        mask_transp: float = 0.5      
    ):

        self.FFT_taps: int = FFT_taps
        
        self.spec_len: int = spec_len

        self.spectrogram = spectrogram

        self.sig_power_t = np.zeros(self.spec_len, dtype = float)

        self.sig_power_f = np.zeros(self.FFT_taps, dtype = float)

        self.noise_power_t = np.zeros(self.spec_len, dtype = float)

        self.noise_power_f = np.zeros(self.FFT_taps, dtype = float)

        self.mask_transp: float = mask_transp

        #self.detected_sig_num: int = 0

        self.detect_thresh_db: float = detect_thresh_db #This is in db. Essentially, if there is a section of the signal at or above this SNR, it is considered detected

        self.detect_thresh: float = 10**(self.detect_thresh_db/10)

        self.detect_mask = np.zeros([self.spec_len, self.FFT_taps,3],dtype = np.uint8)

        self.spec_img_path: string = spec_img_path

    def get_power_t(self):
        
        sum: float = 0

        powers = np.zeros(self.FFT_taps, dtype = float)
        
        for t in range(0,self.spec_len):
            for f in range(0,self.FFT_taps):
                sum += (np.abs(self.spectrogram[t,f]))**2
                powers[f] = (np.abs(self.spectrogram[t,f]))**2
            
            self.sig_power_t[t] = sum/self.FFT_taps
            self.noise_power_t[t] = np.median(powers)

    
    def get_power_f(self):

        sum: float = 0

        powers = np.zeros(self.spec_len, dtype = float)
        
        for f in range(0,self.FFT_taps):
            for t in range(0,self.spec_len):
                sum += (np.abs(self.spectrogram[t,f]))**2
                powers[t] = (np.abs(self.spectrogram[t,f]))**2

            self.sig_power_f[f] = sum/self.spec_len
            self.noise_power_f[f] = np.median(powers)

    
    def find_tx_sigs(self):

        for t in range(0,self.spec_len):
            for f in range(0,self.FFT_taps):
                if ((np.abs(self.spectrogram[t,f]))**2) > self.detect_thresh*self.noise_power_t[t]:
                    self.detect_mask[t,f,0] = 255


    def gen_spec_and_mask(self, detect_mask: np.ndarray, spectrogram: np.ndarray):


        spec_db = np.zeros([spectrogram.shape(0), spectrogram.shape(1)], dtype = float)

        spec_db = 10*np.log10(np.abs(spectrogram) + 1e-12) #add a small amount to avoid any 0's

        spec_db -= spec_db.min() #normalizing values between 0 and 1 to map to integers for print

        spec_db /= spec_db.max()

        spec_img = (255 * spec_db).astype(np.uint8)

        spec_img = Image.fromarray(spec_img, mode="L")

        spec_img_rgb = spec_img.convert("RGB") #this is so we can add the red overlay

        mask_img = Image.fromarray(detect_mask, mode = 'RGB')

        print_img = Image.blend(spec_img_rgb, mask_img, self.mask_transp)

        print_img.save(self.spec_img_path)









