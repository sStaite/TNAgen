from Generator import Generator
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn 


def testGenerator(testing):

    generator = Generator()

    for test in testing:
        generator.generate(test, 32, clean=False)
        #generator.save_as_png("data/sanity_images")

    #generator.save_as_timeseries(path="data/sanity_images", name="test", noise=True, SNR=12)
    #generator.save_as_png("data/sanity_images")
    #generator.save_as_array("data/sanity_images", clear_queue=True)


def testRest():

    filepath = "TNAgen/data/sanity_images/test.gwf"
    strain = TimeSeries.read(filepath, "test")
    white = strain.whiten()

    plot = Plot(strain, white, separate=True, sharex=True, color="forestgreen")
    ax = plot.gca()
    plt.xlim(0, np.ceil(len(strain)/4096))
    plot.refresh()
    plot.savefig("TNAgen/data/sanity_images/comparison.png")
    plot.close()

    # This can be used to display an image of a glitch in spectrogram
    """
    p = 0; i = 0
    while p < 0.25:
        p = glitch_times[i]
        i += 1

    out = (int((p)*4096), int((p+2)*4096))

    NFFT = int(4096/16.)
    NOVL = int(NFFT*15./16)

    window = np.blackman(NFFT)
    spec_cmap='viridis'
    plt.figure(figsize=(8.5, 7))
    spec_H1, freqs, bins, im = plt.specgram(white[out[0]: out[1]], NFFT=NFFT, Fs=4096, window=window,
        noverlap=NOVL, cmap=spec_cmap, scale='linear',mode='magnitude')
    plt.ylim(8, 2048)
    plt.xlabel('time (s)',fontsize=14)
    plt.ylabel('frequency (Hz)',fontsize=14)
    plt.yscale('log', base=2)
    plt.colorbar(cmap=spec_cmap, label="Normalized energy")
    plt.savefig("TNAgen/data/sanity_images/spectrogram_jade.png")
    plt.close()

    qspecgram = np.array(white[out[0]: out[1]].q_transform())
    qspecgram = np.swapaxes(qspecgram, 0, 1)
    
    #print(qspecgram)
    plt.figure(figsize=(8.5, 7))
    plot = seaborn.heatmap(qspecgram, cmap=spec_cmap)
    plot.set(ylim=(8, 2048))
    plot.set_yscale('log', base=2)

    fig = plot.get_figure()
    fig.savefig("TNAgen/data/sanity_images/spectrogram_q.png")
    plt.close()
    """


if __name__ == "__main__":
    all = ["1080Lines", "Helix", "Light_Modulation",
            "Paired_Doves", "Repeating_Blips", "Scattered_Light", "Scratchy", "Violin_Mode", "Whistle", "Wandering_Line",
            "1400Ripples", "Blip", "Chirp", "Koi_Fish", "Tomte", "Air_Compressor", "Power_Line", "Low_Frequency_Burst", "Low_Frequency_Lines"]

    testing = ["Chirp"]

    testGenerator(testing)
    testRest()

    # Low Freq Lines may have less glitches -> this is due to some having too low frequency to be convered into a timeseries -> think this is okay
    # Whistle does not look like actual whistle glitches