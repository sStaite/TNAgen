from Generator import Generator
from gwpy.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def testGenerator(testing):
    generator = Generator()

    for test in testing:
        generator.generate(test, 5, clean=True)
        generator.save_as_png("src/data/sanity_images")
        generator.save_as_hdf5("src/data/sanity_images", test, clear_queue=True)

def testRest(testing):

    for test in testing:
        filepath = f"src/data/sanity_images/{test}.hdf5"

        channel1 = f"{test}_timeseries_0"

        strain = TimeSeries.read(filepath, channel1)
        strain.sample_rate = 4096
        plt.plot(strain, 'forestgreen')
        plt.xlabel("Time (Seconds)")
        plt.savefig(f"src/data/sanity_images/{test}_timeseries.png")
        plt.close()

def testSpectrogram():
    filepath = "src/data/sanity_images/mydata.hdf5"
    channel1 = "Chirp_timeseries_0"

    strain = TimeSeries.read(filepath, channel1)

    fs = 4096

    NFFT = int(fs/16.)
    NOVL = int(NFFT*15./16)

    window = np.blackman(NFFT)
    spec_cmap='viridis'
    plt.figure(figsize=(8.5, 7))
    spec_H1, freqs, bins, im = plt.specgram(strain, NFFT=NFFT, Fs=fs, window=window,
        noverlap=NOVL, cmap=spec_cmap, scale='linear',mode='magnitude')
    plt.ylim(0,4096//2)
    plt.xlabel('time (s)',fontsize=14)
    plt.ylabel('frequency (Hz)',fontsize=14)
    plt.show()

if __name__ == "__main__":
    all = ["1080Lines", "Extremely_Loud", "Helix", "Light_Modulation",
            "Paired_Doves", "Repeating_Blips", "Scattered_Light", "Scratchy", "Violin_Mode", "Wandering_Line", "Whistle",
            "1400Ripples", "Blip", "Chirp", "Koi_Fish", "Tomte", "Air_Compressor", "Power_Line", "Low_Frequency_Burst", "Low_Frequency_Lines"]

    testing = ["1080Lines"]

    testGenerator(testing)
    testRest(testing)
    #testSpectrogram()

    # NO GLITCH AND NONE OF THE ABOVE SHOULD NOT EVEN BE OPTIONS, EXTREMELY LOUD PROBABLY TOO!

    
