import numpy as np

from dsp_demo.OFDM_Gen import OFDM_BPSK, configs

import matplotlib

matplotlib.use("TkAgg")


def main():

    #print("///////////////Testtttttt")
    
    cfg = configs()

    OFDM = OFDM_BPSK(cfg)

    OFDM.msg_gen()

    OFDM.analytic_sig_modulation(cfg)

    OFDM.add_noise(cfg)

    OFDM.gen_spectro(cfg)



if __name__ == "__main__":

    main()