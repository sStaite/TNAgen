from torchgan.models import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import librosa
from scipy import signal
import h5py

#n_images_to_generate = 100
save_images = True

class Generator():
    """
    Transient noise artifacts generator from '"Generating transient noise artefacts in gravitational-wave 
    detector data with generative adversarial networks by Powell et. al. " <https://arxiv.org/abs/2207.00207>' 
    """

    def __init__(self):

        self.curr_array = []
        self.curr_glitch = []

        gen_args = {
            "out_size":256,
            "encoding_dims": 170,
            "out_channels": 1,
            "step_channels": 32,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Sigmoid(),
        }

        self.generator = DCGANGenerator(**gen_args)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            # Use deterministic cudnn algorithms
            torch.backends.cudnn.deterministic = True
        else:
            device = torch.device("cpu")

        # Get the list of glitches
        self.label_df = pd.read_csv('src/data/array_to_label_conversion.csv')
        self.glitches = self.label_df['label'].tolist()


    def generate(self, glitch, n_images_to_generate):
        """
        Generates images for the given glitch in the form of numpy arrays, and adds it to the 'queue'.

        Args:
            glitch (String): Name of the glitch to be generated.
            n_images_to_generate: Number of images to be generated. 

        Returns:
            Returns a tuple: a (n_images_to_generate x 140 x 170) numpy array of the images generated 
            and a (n_images_to_generate) list of corresponding glitch labels.
        
        """

        ## Generate X array for  glitches
        index = 0
        np_array = np.zeros((n_images_to_generate, 140, 170))

        model_weights_file = "src/data/models/{}.model".format(glitch)
        state_dict = torch.load(model_weights_file, map_location='cpu')['generator']
        self.generator.load_state_dict(state_dict)
        
        inputs = [self.generator.sampler(1,torch.device('cpu')) for _ in range(n_images_to_generate)]

        outputs = []
        for i in range(n_images_to_generate):
            outputs.append(self.generator(inputs[i][0]))

        label_list = []

        for i in range(n_images_to_generate):
            im = outputs[i].detach().numpy()
            im = im[0, :,:,:]
            im = im[:, 0:140, 0:170] 
            
            np_array[i] = im
            label_list.append(glitch)

        if len(self.curr_array) == 0:
            self.curr_array = np_array
            self.curr_glitch = label_list
        else:
            self.curr_array = np.concatenate((self.curr_array, np_array), axis=0)
            self.curr_glitch += label_list

        return (np_array, label_list)


    def generate_all(self, n_images_to_generate):
        """
        Generates images for all the glitches, in the form of numpy arrays.

        Args:
            n_images_to_generate: Number of images to be generated, for each glitch

        Returns:
            Returns a tuple: a ((n_images_to_generate * num_of_glitches) x 140 x 170) numpy array of the images generated 
            and a (n_images_to_generate * num_of_glitches) list of corresponding glitch labels.

            The array can be accessed through 'generator.curr_array' and the glitch labels through 'generator.curr_glitch'
        """
        
        np_arrays = np.zeros((len(self.glitches) * n_images_to_generate, 140, 170))
        label_list = []
        index = 0

        for glitch in self.glitches:
            model_weights_file = "src/data/models/{}.model".format(glitch)
            state_dict = torch.load(model_weights_file, map_location='cpu')['generator']
            self.generator.load_state_dict(state_dict)
            
            inputs = [self.generator.sampler(1,torch.device('cpu')) for _ in range(n_images_to_generate)]

            outputs = []
            for i in range(n_images_to_generate):
                outputs.append(self.generator(inputs[i][0]))

            for i in range(n_images_to_generate):
                im = outputs[i].detach().numpy()
                im = im[0, :,:,:]
                im = im[:, 0:140, 0:170] 
                
                np_arrays[index] = im
                label_list.append(glitch)
                index += 1
        

        if len(self.curr_array) == 0:
            self.curr_array = np_arrays
            self.curr_glitch = label_list
        else:
            self.curr_array = np.concatenate((self.curr_array, np_arrays), axis=0)
            self.curr_glitch += label_list

        return (np_arrays, label_list)


    def save_as_png(self, path, clear_queue=False):
        """
        Saves the queue of artifacts, which are in the form of a spectrogram, as png files.

        Args:
            path: Folder where the images will be saved.
            clear_queue: Boolean value for if the queue will be cleared after the images are saved. (Default: False)
        """

        if len(self.curr_array) == 0:
            print("There are currently no generated images.")
            return

        count_dict = {string: 0 for string in set(self.curr_glitch)}

        for i in range(len(self.curr_array)):
            index = count_dict[self.curr_glitch[i]]
            count_dict[self.curr_glitch[i]] += 1
            
            plt.imsave(path + f"/{self.curr_glitch[i]}_{index}.png", self.curr_array[i])
            index+=1

        if clear_queue:
            self.clear_queue()


    def save_as_hdf5(self, path, name="timeseries", clear_queue=False):
        """
        Saves the queue of artifacts in a h5 file. The snippets are 2 seconds long.

        Args:
            path: Folder where the h5 file should be created.
            name (String): Name for the h5 file (Default: "timeseries")
            clear_queue: Boolean value for if the queue will be cleared after the images are saved. (Default: False)
        """

        if len(self.curr_array) == 0:
            print("There are currently no generated images.")
            return

        filepath = path + f"/{name}.hdf5"

        f = h5py.File(filepath, "w")

        for i in range(len(self.curr_array)):
            # First convert the spectrogram data to timeseries data
            time_series = -1 * librosa.griffinlim(self.curr_array[i], n_iter=32)
            time_series = signal.resample(time_series, 8192)
            
            testSpectrogram(time_series)

            """
            plt.plot((np.arange(11661) / float(11661/2)), time_series)
            plt.savefig(path + f"/{self.curr_glitch[i]}_timeseries_{i}.png")
            plt.clf()

            q = librosa.feature.melspectrogram(time_series, 2000)
            print(q.shape)
            plt.pcolormesh(q)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
            """

            f.create_dataset(f"/{self.curr_glitch[i]}_timeseries_{i}", data=time_series, compression="gzip")

        f.flush()
        f.close()

        if clear_queue:
            self.clear_queue()



    def clear_queue(self):
        """
        Clears the current queue of artifacts.
        """
        self.curr_array = []
        self.curr_glitch = []
            



def testGenerator():
    generator = Generator()
    generator.generate("Koi_Fish", 1)
    generator.generate("Blip", 1)
    generator.save_as_png("src/data/images")
    generator.save_as_hdf5("src/data/images", "mydata", clear_queue=True)

from gwpy.timeseries import TimeSeries

def testRest():
    filepath = "src/data/images/mydata.hdf5"
    channel1 = "Koi_Fish_timeseries_0"
    channel2 = "Blip_timeseries_1"

    fs = 4096

    strain = TimeSeries.read(filepath, channel1)

    NFFT = int(fs/16.)
    NOVL = int(NFFT*15./16)

    window = np.blackman(NFFT)

    spec_cmap='gist_ncar'

    plt.figure(figsize=(8,4))

    spec_H1, freqs, bins, im = plt.specgram(strain, NFFT=NFFT, Fs=fs, window=window,

        noverlap=NOVL, cmap=spec_cmap, scale='linear',mode='magnitude')

    plt.ylim(0,2000)

    plt.xlabel('time (s)',fontsize=14)

    plt.ylabel('frequency (Hz)',fontsize=14)

    plt.show()

    """
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
    """

def testSpectrogram(time_series):
    fs = 4096

    f, t, Sxx = signal.spectrogram(time_series, fs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylim(2**3, 2**11.5)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.clf()

    plt.specgram(time_series, NFFT=1024, Fs=fs)
    plt.ylim(2**3, 2**11.5)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.clf()

if __name__ == "__main__":
    testGenerator()
    testRest()


    
