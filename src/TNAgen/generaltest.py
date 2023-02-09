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
        generator.generate(test, 32, clean=True)
        #generator.save_as_png("src/data/sanity_images")

    glitch_times = generator.save_as_timeseries(path="src/data/sanity_images", name="test", noise=True, SNR=12, clear_queue=True)
    return glitch_times


def testRest(testing, glitch_times):

    filepath = "src/data/sanity_images/test.gwf"
    strain = TimeSeries.read(filepath, "test")
    white = strain.whiten()

    plot = Plot(strain, white, separate=True, sharex=True, color="forestgreen")
    ax = plot.gca()
    ax.set_xlim(0, 32)
    plot.refresh()
    plot.savefig("src/data/sanity_images/comparison.png")
    plot.close()

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
    plt.savefig("src/data/sanity_images/spectrogram_jade.png")
    plt.close()

    qspecgram = np.array(white[out[0]: out[1]].q_transform())
    qspecgram = np.swapaxes(qspecgram, 0, 1)
    
    #print(qspecgram)
    plt.figure(figsize=(8.5, 7))
    plot = seaborn.heatmap(qspecgram, cmap=spec_cmap)
    plot.set(ylim=(8, 2048))
    plot.set_yscale('log', base=2)

    fig = plot.get_figure()
    fig.savefig("src/data/sanity_images/spectrogram_q.png")
    plt.close()

    """
    plt.plot(strain, 'forestgreen')
    plt.xlabel("Time (Seconds)")
    plt.savefig("src/data/sanity_images/timeseries.png")
    plt.xlim(1.85, 2)
    plt.savefig("src/data/sanity_images/timeseries_short.png")
    plt.close()
    """

def testSpectrogram():
    filepath = "src/data/sanity_images/test.gwf"

    strain = TimeSeries.read(filepath)

    fs = 4096

    NFFT = int(fs/16.)
    NOVL = int(NFFT*15./16)

    window = np.blackman(NFFT)
    spec_cmap='viridis'
    
def test_array_save():
    generator = Generator()

    for test in testing:
        generator.generate(test, 3, clean=True)

    generator.save_as_array("src/data/sanity_images", "test")


if __name__ == "__main__":
    all = ["1080Lines", "Extremely_Loud", "Helix", "Light_Modulation",
            "Paired_Doves", "Repeating_Blips", "Scattered_Light", "Scratchy", "Violin_Mode", "Wandering_Line", "Whistle",
            "1400Ripples", "Blip", "Chirp", "Koi_Fish", "Tomte", "Air_Compressor", "Power_Line", "Low_Frequency_Burst", "Low_Frequency_Lines"]

    testing = ["Chirp", "Blip", "Koi_Fish"]

    #g = testGenerator(testing)
    #testRest(testing, g)
    #testSpectrogram()
    test_array_save()

    
