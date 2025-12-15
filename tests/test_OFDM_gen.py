import numpy as np

from dsp_demo.OFDM_Gen import OFDM, configs

import matplotlib

matplotlib.use("TkAgg")


def main():

    #print("///////////////Testtttttt")
    
    cfg = configs()

    OFDM_loc = OFDM(cfg)

    OFDM_loc.msg_gen()

    #OFDM_loc.analytic_sig_BPSK_mod(cfg)

    OFDM_loc.IFFT_sig_BPSK_mod(cfg)

    OFDM_loc.add_noise(cfg)

    OFDM_loc.gen_spectro(cfg)



if __name__ == "__main__":

    main()