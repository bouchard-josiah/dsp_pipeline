import numpy as np

from dsp_demo.OFDM_Gen import OFDM, configs

from dsp_demo.Energy_Detection import Energy_Detect

#from PIL import IMAGE


def main():

    #print("///////////////Testtttttt")
    
    cfg = configs()

    OFDM_loc = OFDM(cfg)

    OFDM_loc.msg_gen()

    OFDM_loc.IFFT_sig_BPSK_mod(cfg)

    OFDM_loc.add_noise(cfg)

    OFDM_loc.gen_spectro(cfg)

    Energy_loc = Energy_Detect(spectrogram=OFDM_loc.spectrogram, FFT_taps=cfg.FFT_taps, spec_len=cfg.spec_len, spec_img_path=cfg.spec_location)

    Energy_loc.get_power_f()

    Energy_loc.get_power_t()

    Energy_loc.find_tx_sigs()

    Energy_loc.gen_spec_and_mask()



if __name__ == "__main__":

    main()