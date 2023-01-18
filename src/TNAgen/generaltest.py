from Generator import Generator
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def testGenerator():
    generator = Generator()
    generator.generate("Koi_Fish", 1)
    generator.generate("Blip", 1)
    generator.save_as_png("src/data/images")
    generator.save_as_hdf5("src/data/images", "mydata", clear_queue=True)

def testRest():
    filepath = "src/data/images/mydata.hdf5"
    channel1 = "Koi_Fish_timeseries_0"
    channel2 = "Blip_timeseries_1"

    strain = TimeSeries.read(filepath, channel1)
    strain.sample_rate = 4096
    plt.plot(strain, 'forestgreen')
    plt.xlabel("Time (Seconds)")
    plt.savefig("src/data/images/test1.png")
    plt.clf()

    strain = TimeSeries.read(filepath, channel2)
    strain.sample_rate = 4096
    plt.plot(strain, 'plum')
    plt.xlabel("Time (Seconds)")
    plt.savefig("src/data/images/test2.png")
    plt.clf()
    plt.close()

def testSpectrogram():
    filepath = "src/data/images/mydata.hdf5"
    channel1 = "Koi_Fish_timeseries_0"

    strain = TimeSeries.read(filepath, channel1)

    fs = 4096

    NFFT = int(fs/16.)
    NOVL = int(NFFT*15./16)

    window = np.blackman(NFFT)
    spec_cmap='gist_ncar'
    plt.figure(figsize=(7, 8.5))
    spec_H1, freqs, bins, im = plt.specgram(strain, NFFT=NFFT, Fs=fs, window=window,
        noverlap=NOVL, cmap=spec_cmap, scale='linear',mode='magnitude')
    plt.ylim(0,4096//2)
    plt.xlabel('time (s)',fontsize=14)
    plt.ylabel('frequency (Hz)',fontsize=14)
    plt.show()

if __name__ == "__main__":
    testGenerator()
    testRest()
    testSpectrogram()

    
