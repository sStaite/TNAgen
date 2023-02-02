from torchgan.models import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import librosa
from scipy import signal, interpolate
import h5py
import cv2

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
        self.PSD = np.swapaxes(np.loadtxt('src/data/ALIGO_noise_curve.txt'), 0, 1)


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
                im = self._clean_spectrogram(im, glitch)

            np_array[i] = im 

            label_list.append(glitch)

            self._timer("Generating images:", i+1, n_images_to_generate)
        
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
                    im = self._clean_spectrogram(im, glitch)

                np_arrays[index] = im
                label_list.append(glitch)

                self._timer("Generating images:", index+1, n_images_to_generate*len(self.glitches))
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

            self._timer("Saving images:    ", i+1, len(self.curr_array))

            index+=1

        if clear_queue:
            self._clear_queue()


    def save_as_timeseries(self, path, name="timeseries", noise=True, length="Default", position=None, SNR=10, clear_queue=False):
        """
        Saves the queue of artifacts in a h5 file. The snippets are 1/3 * num of glitches seconds long, unless specified otherwise.

        Args:
            path: Folder where the h5 file should be created.
            name (String): Name for the h5 file (Default: "timeseries")
            noise: If the user wants noise to be saved with the timeseries
            length: The length (in seconds) of the snippet. Default is 1/3 * the number of glitches given.
            position: The positions of the start of glitches, given in the form of a numpy array in seconds; must be of size len(self.glitches)
                        Default=None: The glitches will be distributed randomly 
            SNR: The Signal to noise ratio that is wanted for the glitches. (Default: 10)
            clear_queue: Boolean value for if the queue will be cleared after the images are saved. (Default: False)
        """

        # Check there are glitches generated.
        if len(self.curr_array) == 0:
            print("There are currently no generated glitches.")
            return
        
        if position is not None and (len(self.curr_array) != len(position)):
            print("Position array is not the same length as the number of glitches.")
            return

        filepath = path + f"/{name}.hdf5"

        f = h5py.File(filepath, "w")

        # Create a timeseries which just has gaussian noise for our specific PSD
        if length == "Default":
            length = np.ceil(len(self.curr_array)*4096 / 3) / 4096 

        b = np.arange(start=0, stop=length, step=(1/4096))

        if noise:
            n = self._gaussian_data(length, self.PSD)
        else: 
            n = np.zeros(int(length * 4096)) 

        timeseries = np.array(list(zip(b, n)))

        count_dict = {string: 0 for string in set(self.curr_glitch)}
        for i in range(len(self.curr_array)):
            # First convert the spectrogram data to timeseries data
            index = count_dict[self.curr_glitch[i]]
            count_dict[self.curr_glitch[i]] += 1
            curr_time_series = self._convert_to_timeseries(self.curr_array[i])

            # Then we need to adjust the amplitude of the timeseries for the required SNR.
            curr_time_series = self._adjust_amplitude(curr_time_series, self.PSD, SNR)

            # Find the starting position of the glitch
            if position is None:
                curr_pos = np.random.choice(np.arange(len(timeseries[:, 0])) - len(curr_time_series) // 2)
            else:
                curr_pos = position[i]
            
            # Add the glitch in
            # This code just lets the data from the glitch be put anywhere on the timeseries
            for i in range(len(curr_time_series)):
                if (curr_pos + i >= length * 4096):
                    break
                if (curr_pos + i >= 0):
                    timeseries[curr_pos + i, 1] += curr_time_series[i]


            #self.__timer("Saving timeseries:", i+1, len(self.curr_array))
            index+=1

        # Save dataset
        timeseries = np.swapaxes(timeseries, 0, 1)[1]
        f.create_dataset(f"/timeseries", data=timeseries, compression="gzip")
        f.flush()
        f.close()

        if clear_queue:
            self._clear_queue()


    def _convert_to_timeseries(self, spectrogram):
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


    def _clean_spectrogram(self, spectrogram, glitch):
        """
        Removes all datapoints below a certain threshold.

        Args:
            spectrogram: a 170x140 numpy array of spectrogram data 

        Returns:
            A spectrogram of the same shape, which has been cleaned of noise.
        """


        # First, get rid of all background noise
        both = ["Paired_Doves", "Extremely_Loud", "Air_Compressor", "Low_Frequency_Lines", "1400Ripples", "Blip", "Chirp", "Koi_Fish", "Tomte", "Power_Line"]
        recurring_horizontally = ["Scattered_Light", "Wandering_Line", "Violin_Mode"]
        recurring_vertically = ["1080Lines", "Low_Frequency_Burst", "Repeating_Blips",  "Scratchy", "Whistle"]
        neither = ["Light_Modulation", "Helix", "Whistle"]
        if glitch in both: 
            threshold = 0.30
            if glitch in ["Low_Frequency_Lines", "Paired_Doves"]: 
                threshold = 0.25

            self._clean_vertically(spectrogram, threshold)
            self._clean_horizontally(spectrogram, threshold)
            spectrogram[spectrogram < threshold] = 0

        if glitch in recurring_horizontally:
            threshold = 0.30

            if glitch == "Violin_Mode":
                spectrogram[0, 50:, :] = 0
                threshold = 0.25

            self._clean_vertically(spectrogram, threshold)
            spectrogram[spectrogram < threshold] = 0

        if glitch in recurring_vertically:
            threshold = 0.30
            self._clean_horizontally(spectrogram, threshold)
            spectrogram[spectrogram < threshold] = 0
        
        if glitch in neither: 
            threshold = 0.3

            if glitch == "Whistle":
                threshold = 0.25

            spectrogram[spectrogram < threshold] = 0
        
        return spectrogram


    def _clean_vertically(self, spectrogram, threshold):
        """
        Helper function to clean_spectrogram that removes all extra noise from the left and right hand sides of the glitch (cleaning columns)
        """
        indmax = np.unravel_index(np.argmax(spectrogram, axis=None), spectrogram.shape)
        min_time = max_time = indmax[2]
        p = spectrogram[indmax]
            
        # Find the upper and lower bounds of the glitch
        while (p > threshold):
            if spectrogram[0, :, min_time].max() > threshold:
                min_time -= 1
                
            if spectrogram[0, :, max_time].max() > threshold:
                max_time += 1

            if min_time == -1:
                min_time = 0
                min_max = 0
            else: 
                min_max = spectrogram[0, :, min_time].max()

            if max_time == 170:
                max_time = 169
                max_max = 0
            else:
                max_max = spectrogram[0, :, max_time].max()

            p = max(min_max, max_max)
        
        spectrogram[0, :, 0:min_time] = 0
        spectrogram[0, :, max_time:] = 0    


    def _clean_horizontally(self, spectrogram, threshold):
        """
        Helper function to clean_spectrogram that removes all extra noise from the top and bottom of the glitch (cleaning rows)
        """
        indmax = np.unravel_index(np.argmax(spectrogram, axis=None), spectrogram.shape)
        min_time = max_time = indmax[1]
        p = spectrogram[indmax]
            
            # Find the upper and lower bounds of the glitch
        while (p > threshold):
            if spectrogram[0, min_time, :].max() > threshold:
                min_time -= 1
                
            if spectrogram[0, max_time, :].max() > threshold:
                max_time += 1


            if min_time == -1:
                min_time = 0
                min_max = 0
            else:
                min_max = spectrogram[0, min_time, :].max()

            if max_time == 140:
                max_time = 139
                max_max = 0
            else: 
                max_max = spectrogram[0, max_time, :].max()



            p = max(min_max, max_max)
            
        spectrogram[0, 0:min_time, :] = 0
        spectrogram[0, max_time:, :] = 0
    
    
    def _clear_queue(self):
        """
        Clears the current queue of artifacts.
        """
        self.curr_array = []
        self.curr_glitch = []
            
    
    def _timer(self, msg, curr, total):
        bars = curr*20 // total
        digits = len(str(total))
        print(msg + " " + str(curr).rjust(digits, " ") + f"/{total} [" + "-"*bars + " "*(20-bars) + "]", end="\r")

        if curr == total:
            print()
    

    def _adjust_amplitude(self, timeseries, PSD, requiredSNR):
        """
        Calculate's the Signal to noise ratio for a given glitch.
        """

        fs = 4096

        amp = 1e-20 # Give the glitch a realistic amplitude
        timeseries *= amp

        # Convert the timeseries into freqsignal
        freq_signal = np.fft.rfft(timeseries) / fs 
        freq_values = np.fft.fftfreq(len(timeseries), d=1./fs)
        
        # Get only the frequencies between 10Hz and 2048Hz
        freq_signal = freq_signal[20:4096]
        freq_values = freq_values[20:4096]

        # Resample PSD  
        f = interpolate.interp1d(PSD[0], PSD[1])
        new_psd = f(freq_values)

        # Find the frequency spacing
        df = freq_values[1] - freq_values[0]

        # Calculate current SNR 
        SNRsq = 4 * df * np.sum(pow(abs(freq_signal),2.)/ new_psd)
        SNR = np.sqrt(SNRsq)

        # Since all calculations are linear, we can just adjust timeseries to get SNR we need
        timeseries *= requiredSNR/SNR 

        return timeseries

      
    def _gaussian_data(self, PSD):
        return 0

