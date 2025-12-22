import numpy as np

from dsp_demo.OFDM_Gen import OFDM, configs

from dsp_demo.FIR_Filter import Filters

import string

from pathlib import Path



def main():
        
    cfg = configs()

    OFDM_loc = OFDM(cfg)

    OFDM_loc.msg_gen()

    OFDM_loc.IFFT_sig_BPSK_mod(cfg)

    OFDM_loc.add_noise(cfg)

    OFDM_loc.gen_spectro(cfg)

    OFDM_loc.print_spectro(cfg,spectrogram = OFDM_loc.spectrogram, spec_location=OFDM_loc.spec_location)

    filter_bw: float = float((OFDM_loc.msg_carriers*1.2)*cfg.scs)

    Filter_loc = Filters(spectrogram=OFDM_loc.spectrogram, FFT_taps=cfg.FFT_taps, freq_offset = OFDM_loc.f_offset, sample_rate = OFDM_loc.sample_rate, filter_bw = filter_bw)

    Filter_loc.overlap_add_filter()

    spec_location: string = str(Path.home()) + "/dsp_demo/results/spectrogram6.png"

    OFDM_loc.print_spectro(cfg,spectrogram = Filter_loc.filtered_sig_f, spec_location=spec_location)


if __name__ == "__main__":

    main()