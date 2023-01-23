from torchgan.models import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import librosa
from scipy import signal, interpolate
import h5py

#n_images_to_generate = 100
save_images = True

class Generator():
    """
    Transient noise artifacts generator from '"Generating transient noise artefacts in gravitational-wave 
    detector data with generative adversarial networks by Powell et. al. " <https://arxiv.org/abs/2207.00207>' 
    """

    def __init__(self):

        # Initialize empty lists to store generated images and labels
        self.curr_array = []
        self.curr_glitch = []

        # Set up the generator  parameters
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


    def generate(self, glitch, n_images_to_generate, clean=False):
        """
        Generates images for the given glitch in the form of numpy arrays, and adds it to the 'queue'.

        Args:
            glitch (String): Name of the glitch to be generated.
            n_images_to_generate: Number of images to be generated. 

        Returns:
            Returns a tuple: a (n_images_to_generate x 140 x 170) numpy array of the images generated 
            and a (n_images_to_generate) list of corresponding glitch labels.
        
        """

        ## Generate X array for glitches
        index = 0
        np_array = np.zeros((n_images_to_generate, 140, 170))

        # Load model weights for the specified glitch
        model_weights_file = "src/data/models/{}_GAN.model".format(glitch)
        state_dict = torch.load(model_weights_file, map_location='cpu')
        self.generator.load_state_dict(state_dict)
        
        # Generate the images using the generator
        inputs = []
        outputs = []
        label_list = []
        for i in range(n_images_to_generate):
            inputs.append(self.generator.sampler(1,torch.device('cpu')))
            
            outputs.append(self.generator(inputs[i][0]))

            # Turn output into numpy arrays
            im = outputs[i].detach().numpy()
            im = im[0, :,:,:]
            im = im[:, 0:140, 0:170] 
            
            if clean:
                im = self.clean_spectrogram(im, glitch)

            np_array[i] = im 

            label_list.append(glitch)

            self.__timer("Generating images:", i+1, n_images_to_generate)
        
        # Update the queue or create it if it does not exist
        if len(self.curr_array) == 0:
            self.curr_array = np_array
            self.curr_glitch = label_list
        else:
            self.curr_array = np.concatenate((self.curr_array, np_array), axis=0)
            self.curr_glitch += label_list

        return (np_array, label_list)


    def generate_all(self, n_images_to_generate, clean=False):
        """
        Generates images for all the glitches, in the form of numpy arrays.

        Args:
            n_images_to_generate: Number of images to be generated, for each glitch

        Returns:
            Returns a tuple: a ((n_images_to_generate * num_of_glitches) x 140 x 170) numpy array of the images generated 
            and a (n_images_to_generate * num_of_glitches) list of corresponding glitch labels.

            The array can be accessed through 'generator.curr_array' and the glitch labels through 'generator.curr_glitch'
        """
        
        # Generate X array for glitches
        np_arrays = np.zeros((len(self.glitches) * n_images_to_generate, 140, 170))
        label_list = []
        index = 0

        # Iterate over each glitch and generate the images
        for glitch in self.glitches:
            model_weights_file = "src/data/models/{}_GAN.model".format(glitch)
            state_dict = torch.load(model_weights_file, map_location='cpu')
            self.generator.load_state_dict(state_dict)
            
            inputs = []
            outputs = []
            for i in range(n_images_to_generate):
                inputs.append(self.generator.sampler(1,torch.device('cpu')))
                
                outputs.append(self.generator(inputs[i][0]))

                # Turn output into numpy arrays
                im = outputs[i].detach().numpy()
                im = im[0, :,:,:]
                im = im[:, 0:140, 0:170] 

                if clean:
                    im = self.clean_spectrogram(im, glitch)

                np_arrays[index] = im
                label_list.append(glitch)

                self.__timer("Generating images:", index+1, n_images_to_generate*len(self.glitches))
                index+=1


        
        # Update the queue or create it if it does not exist
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

            self.__timer("Saving images:    ", i+1, len(self.curr_array))

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

        count_dict = {string: 0 for string in set(self.curr_glitch)}

        for i in range(len(self.curr_array)):
            # First convert the spectrogram data to timeseries data

            index = count_dict[self.curr_glitch[i]]
            count_dict[self.curr_glitch[i]] += 1

            time_series = self.convert_to_timeseries(self.curr_array[i]) * (1/0.04)

            self.__timer("Saving timeseries:", i+1, len(self.curr_array))

            f.create_dataset(f"/{self.curr_glitch[i]}_timeseries_{index}", data=time_series, compression="gzip")

            index+=1

        f.flush()
        f.close()

        if clear_queue:
            self.clear_queue()


    def convert_to_timeseries(self, spectrogram):
        """
        Converts the given spectrogram into a timeseries.

        Args:
            spectrogram: a 170x140 numpy array of a spectrogram of a glitch
    
        Returns:
            A 2 second timeseries (8192 datapoints at 4096Hz) of the spectrogram data
        """

        fs = 4096
        NFFT = int(fs/16.)
        NOVL = int(NFFT*15./16)

        time_series = librosa.griffinlim(spectrogram, n_iter=64)
        time_series[1::2] *= -1            
        time_series = signal.resample(time_series, 8192)

        return time_series


    def clean_spectrogram(self, spectrogram, glitch):
        """
        Removes all datapoints below a certain threshold.

        Args:
            spectrogram: a 170x140 numpy array of spectrogram data 

        Returns:
            A spectrogram of the same shape, which has been cleaned of noise.
        """

        leftover = ["1080Lines", "Extremely_Loud", "Helix", "Light_Modulation", "No_Glitch", "None_of_the_Above", 
                    "Paired_Doves", "Repeating_Blips", "Scattered_Light", "Scratchy", "Violin_Mode", "Wandering_Line", "Whistle"]
        both = ["Air_Compressor", "Power_Line", "Low_Frequency_Burst", "Low_Frequency_Lines"]
        vertical_only = ["1400Ripples", "Blip", "Chirp", "Koi_Fish", "Tomte"]

        if (glitch in vertical_only):
            
            threshold = 0.3

            indmax = np.unravel_index(np.argmax(spectrogram, axis=None), spectrogram.shape)
            min_time = max_time = indmax[2]
            p = spectrogram[indmax]
            
            # Find the upper and lower bounds of the glitch
            while (p > threshold):

                if spectrogram[0, :, min_time].max() > threshold:
                    min_time -= 1
                
                if spectrogram[0, :, max_time].max() > threshold:
                    max_time += 1

                if min_time == -1 or max_time == 170:
                    min_time = 0
                    max_time = 169
                    break

                p = max(spectrogram[0, :, min_time].max(), spectrogram[0, :, max_time].max())
            

            spectrogram[0, :, 0:min_time] = 0
            spectrogram[0, :, max_time:] = 0
            spectrogram[spectrogram < threshold] = 0

        return spectrogram
    
    
    def clear_queue(self):
        """
        Clears the current queue of artifacts.
        """
        self.curr_array = []
        self.curr_glitch = []
            
    
    def __timer(self, msg, curr, total):
        bars = curr*20 // total
        digits = len(str(total))
        print(msg + " " + str(curr).rjust(digits, " ") + f"/{total} [" + "-"*bars + " "*(20-bars) + "]", end="\r")

        if curr == total:
            print()