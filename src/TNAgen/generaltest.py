from Generator import Generator
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def testGenerator(testing):
    generator = Generator()

    for test in testing:
        generator.generate(test, 96, clean=True)
        #generator.save_as_png("src/data/sanity_images")

    glitch_times = generator.save_as_timeseries(path="src/data/sanity_images", name="test", noise=True, SNR=10, clear_queue=True)
    return glitch_times


def testRest(testing, glitch_times):

    filepath = "src/data/sanity_images/test.gwf"
    strain = TimeSeries.read(filepath, "test")
    white = strain.whiten()

    print(strain)

    fig2 = strain.asd().plot()
    plt.xlim(2**3,2**11)
    plt.ylim(1e-25, 1e-19)
    plt.savefig("src/data/sanity_images/asd.png")
    plt.close()

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
    spec_H1, freqs, bins, im = plt.specgram(strain[out[0]: out[1]], NFFT=NFFT, Fs=4096, window=window,
        noverlap=NOVL, cmap=spec_cmap, scale='linear',mode='magnitude')
    plt.ylim(10, 2000)
    plt.xlabel('time (s)',fontsize=14)
    plt.ylabel('frequency (Hz)',fontsize=14)
    plt.yscale('log', base=2)
    plt.savefig("src/data/sanity_images/spectrogram_jade.png")


    """
    plt.plot(strain, 'forestgreen')
    plt.xlabel("Time (Seconds)")
    plt.savefig("src/data/sanity_images/timeseries.png")
    plt.xlim(1.85, 2)
    plt.savefig("src/data/sanity_images/timeseries_short.png")
    plt.close()
    """

def testSpectrogram():
    filepath = "src/data/sanity_images/timeseries.hdf5"

    strain = TimeSeries.read(filepath)

    fs = 4096

    NFFT = int(fs/16.)
    NOVL = int(NFFT*15./16)

    window = np.blackman(NFFT)
    spec_cmap='viridis'
    plt.figure(figsize=(8.5, 7))
    spec_H1, freqs, bins, im = plt.specgram(strain, NFFT=NFFT, Fs=fs, window=window,
        noverlap=NOVL, cmap=spec_cmap, scale='linear',mode='magnitude')
    plt.ylim(10,4096//2)
    plt.xlabel('time (s)',fontsize=14)
    plt.ylabel('frequency (Hz)',fontsize=14)
    plt.savefig("src/data/sanity_images/spectrogram_jade.png")

if __name__ == "__main__":
    all = ["1080Lines", "Extremely_Loud", "Helix", "Light_Modulation",
            "Paired_Doves", "Repeating_Blips", "Scattered_Light", "Scratchy", "Violin_Mode", "Wandering_Line", "Whistle",
            "1400Ripples", "Blip", "Chirp", "Koi_Fish", "Tomte", "Air_Compressor", "Power_Line", "Low_Frequency_Burst", "Low_Frequency_Lines"]

    testing = ["Blip"]

    g = testGenerator(testing)
    testRest(testing, g)
    #testSpectrogram()

    # NO GLITCH AND NONE OF THE ABOVE SHOULD NOT EVEN BE OPTIONS, EXTREMELY LOUD PROBABLY TOO!

    
