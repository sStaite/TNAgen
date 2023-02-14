from torchgan.models import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import librosa
from scipy import signal, interpolate
from gwpy.timeseries import TimeSeries
import h5py
import framel

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

        # Get the standard PSD
        self.PSD = np.swapaxes(np.loadtxt('src/data/ALIGO_noise_curve.txt'), 0, 1)


    def generate(self, glitch, n_images_to_generate=10, clean=True):
        """
        Generates images for the given glitch in the form of numpy arrays, and adds it to the 'queue'.

        :param glitch: Name of the glitch to be generated
        :type glitch: str
        :param n_images_to_generate: Number of images to be generated, defaults to 10
        :type n_images_to_generate: int, optional 
        :param clean: Whether or not to remove the background noise from the generated glitches, defaults to True
        :type clean: bool, optional
        :return: A (n_images_to_generate x 140 x 170) numpy array of the images generated 
            and a (n_images_to_generate) list of corresponding glitch labels
        :rtype: tuple
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


    def generate_all(self, n_images_to_generate=1, clean=True):
        """
        Generates images for all the glitches, in the form of numpy arrays. The generated array can be accessed 
        through 'generator.curr_array' and the glitch labels through 'generator.curr_glitch'.

        :param n_images_to_generate: Number of each image to be generated, defaults to 1
        :type n_images_to_generate: int, optional
        :param clean: Whether or not to remove the background noise from the generated glitches, defaults to True
        :type clean: bool, optional
        :return: a ((n_images_to_generate * num_of_glitches) x 140 x 170) numpy array of the images generated 
            and a (n_images_to_generate * num_of_glitches) list of corresponding glitch labels.
        :rtype: tuple
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
        Saves the queue of artifacts, which are in the form of a spectrogram, as seperate png files.

        :param path: Folder where the images will be saved
        :type path: str
        :param clear_queue: Value for if the queue will be cleared after the images are saved, defaults to False
        :type clear_queue: bool, optional
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


    def save_as_timeseries(self, path, name="timeseries", noise=True, length="Default", position=None, SNR=10, format="gwf", clear_queue=False):
        """
        Saves the queue of artifacts in a file. The snippets are 1/3 * num of glitches seconds long, unless specified otherwise.
        Sample rate and the position of the glitches are saved into the file. The channel name of the timeseries is the same as the name of the file.

        :param path: Folder where the timeseries file will be created
        :type path: str
        :param name: name of the file and the channel in the file, defaults to "timeseries"
        :type name: str, optional
        :param noise: Whether or not gaussian noise should be added to the timeseries (Note - for best results, set clean=True in the generation of the glitches
         if noise=True is wanted), defaults to True
        :type noise: bool, optional
        :param length: The length (in seconds) of the snippet. Default is 1/3 * the number of glitches given, defaults to "Default" 
        :type length: float, optional
        :param position: The positions of the start of glitches and must be a list or numpy array of size len(self.glitches), with the positons give in seconds, defaults to None
        :type position: list, optional
        :param SNR: The signal to noise ratio for the glitches, defaults to 10
        :type SNR: float, optional
        :param format: Name of the format that the file will be saved as with pptions "gwf", "hdf5", defaults to "gwf"
        :type format: str, optional
        :param clear_queue: Value for if the queue will be cleared after the images are saved, defaults to False
        :type clear_queue: bool, optional
        """

        # Check there are glitches generated.
        if len(self.curr_array) == 0:
            raise Exception("There are currently no generated glitches.")
        
        if position is not None and (len(self.curr_array) != len(position)):
            raise Exception("Position array is not the same length as the number of glitches.")

        # Create a timeseries which just has gaussian noise for our specific PSD
        if length == "Default":
            length = np.ceil(len(self.curr_array)*4096 / 3) / 4096 

        b = np.arange(start=0, stop=length, step=(1/4096))
        n = np.zeros(int(length * 4096)) 
        timeseries = np.array(list(zip(b, n)))
        glitch_times = []

        count_dict = {string: 0 for string in set(self.curr_glitch)}
        for i in range(len(self.curr_array)):
            # First convert the spectrogram data to timeseries data
            index = count_dict[self.curr_glitch[i]]
            count_dict[self.curr_glitch[i]] += 1
            curr_time_series = self._convert_to_timeseries(self.curr_array[i])

            # Then we need to adjust the amplitude of the timeseries for the required SNR.
            curr_time_series = self._adjust_amplitude(curr_time_series, self.PSD, SNR)

            # Check that the time series exists
            if curr_time_series is None:
                continue

            # Find the starting position of the glitch
            if position is None:
                curr_pos = np.random.choice(np.arange(len(timeseries[:, 0])) - len(curr_time_series) // 2)
            else:
                curr_pos = position[i]

            glitch_times += [curr_pos / 4096]

            # Add the glitch in
            # This code just lets the data from the glitch be put anywhere on the timeseries
            for j in range(len(curr_time_series)):
                if (curr_pos + j >= length * 4096):
                    break
                if (curr_pos + j >= 0):
                    timeseries[curr_pos + j, 1] += curr_time_series[j]


            self._timer("Saving timeseries:", i+1, len(self.curr_glitch))
            index+=1

        # Save dataset
        timeseries = np.swapaxes(timeseries, 0, 1)[1]

        if noise:
            timeseries = self._add_gaussian_noise(timeseries, self.PSD, duration=length) 

        # Save the dataset
        if format == "gwf":
            t = TimeSeries(timeseries, sample_rate=4096, name=f'{name}', channel="CHANNEL")
            t.write(path + f"/{name}.gwf")
        elif format == "hdf5":
            filepath = path + f"/{name}.hdf5"
            f = h5py.File(filepath, "w")
            f.create_dataset(f"/{name}", data=timeseries, compression="gzip")
            f.flush()
            f.close()
        else:
            raise Exception(f"The format {format} cannot be used.")

        if clear_queue:
            self._clear_queue()


    def save_as_array(self, path, name="glitch_file", clear_queue=False):
        """
        Saves the queue of artifacts, which are in the form of a spectrogram - in their basic array form. The 2d arrays, which are 170x140, 
        are saved into a hdf5 file. Each glitch type has its own dataset - within each of these datasets there is a Nx170x140 array, for a N of a specific glitch.  

        :param path: Folder where the timeseries file will be created
        :type path: str
        :param name: name of the file, defaults to "glitch_file"
        :type name: str, optional
        :param clear_queue: Value for if the queue will be cleared after the images are saved, defaults to False
        :type clear_queue: bool, optional
        """

        f = h5py.File(path + f"/{name}.hdf5", "w")
        glitches_used = list(set(self.curr_glitch))
        num_glitches = len(glitches_used)

        dct = {glitches_used[i]: np.empty(shape=(0, 140, 170)) for i in range(num_glitches)}

        for i in range(len(self.curr_glitch)):
            o = dct[self.curr_glitch[i]]
            n = self.curr_array[i].reshape(1, 140, 170)
            dct[self.curr_glitch[i]] = np.concatenate((o, n))

            self._timer("Saving array     :", i+1, len(self.curr_glitch))


        for g in glitches_used:
            f.create_dataset(g, data=dct[g], compression="gzip")

        f.flush()
        f.close()

        if clear_queue:
            self._clear_queue()


    def _convert_to_timeseries(self, spectrogram):
        """
        Helper function that converts the given spectrogram into a timeseries.

        :param spectrogram: a 170x140 numpy array of a spectrogram of a glitch
        :type spectrogram: numpy array
        :return: A 2 second timeseries (8192 datapoints at 4096Hz) of the spectrogram data
        :rtype: numpy array
        """

        freq_values = np.logspace(3, 11, 140, base=2)
        arr = np.linspace(8, 2048, 140)
        
        f = interpolate.interp1d(arr, spectrogram[:, 0])
        spec = np.zeros((1, 140))
        spec[0] = np.array(f(freq_values))

        # normalise the spectrogram, but also flip the data (this is what we want for the timeseries!)
        for i in range(169):
            f = interpolate.interp1d(arr, spectrogram[:, i+1])
            new = np.zeros((1, 140))
            new[0] = np.array(f(freq_values))
            spec = np.concatenate((spec, new))
        
        spec = np.swapaxes(spec, 0, 1)
        #spec = np.flip(spec, axis=0)

        """
        f, ax = plt.subplots(figsize=(8.5, 7))
        ax.imshow(spec)
        ax.set_title("Spectrogram Undistorted", size=20)
        ax.invert_yaxis()
        ax.set_yticks(ticks = np.arange(0, 140, 140/6), labels = np.arange(8, 2048, 340))
        ax.set_xticks(ticks = np.arange(0, 170, 170/8), labels = np.arange(0, 2, 0.25))
        plt.savefig("src/data/sanity_images/specundistort.png")     
        plt.close()  
        """        

        time_series = librosa.griffinlim(spec, n_iter=64)
        time_series[1::2] *= -1            
        time_series = signal.resample(time_series, 8192)

        return time_series


    def _clean_spectrogram(self, spectrogram, glitch):
        """
        Helper function that removes all datapoints from a spectrogram below a certain threshold. At the moment the threshold is
        manually found, but in the future it would be good to do this automatically. 

        :param spectrogram: a 170x140 numpy array of spectrogram data 
        :type spectrogram: numpy array
        :param glitch: name of the glitch that is being cleaned
        :type glitch: str
        :return: a spectrogram of the same shape that has been cleaned
        :rtype: numpy array
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

        :param spectrogram: a 170x140 numpy array of spectrogram data 
        :type spectrogram: numpy array
        :param threshold: Threshold for which pixels should be cleaned and which should be untouched
        :type threshold: float
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

        :param spectrogram: a 170x140 numpy array of spectrogram data 
        :type spectrogram: numpy array
        :param threshold: Threshold for which pixels should be cleaned and which should be untouched
        :type threshold: float
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
        Helper function that calculate's the signal to noise ratio for a given glitch.

        :param timeseries: The timeseries of the glitch
        :type timeseries: numpy array
        :param PSD: The PSD that is used to calculate the SNR.
        :type PSD: numpy array
        :param requiredSNR: the SNR that the user wants the glitch to be.
        :type requiredSNR: float
        :return: The timeseries of the glitch, which has had its amplitude adjusted to match the SNR that is desired.
        :rtype: numpy array
        """

        fs = 4096

        amp = 1e-20 # Give the glitch a realistic amplitude
        timeseries *= amp

        # Convert the timeseries into freqsignal
        freq_signal = np.fft.rfft(timeseries) / fs 

        # If our frequency signal is not produced correctly, return None
        if (np.sum(freq_signal) == 0):
            return None 

        freq_values = np.fft.fftfreq(len(timeseries), d=1./fs)
        
        # Get only the frequencies between 8Hz and 2048Hz
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

      
    def _add_gaussian_noise(self, timeseries, PSD, duration):
        """
        Helper function that adds gaussian noise to the given timeseries.

        :param timeseries: The timeseries that is noise is added to
        :type timeseries: numpy array
        :param PSD: The PSD that is used to generate the gaussian noise, so it fits with the timeseries.
        :type PSD: numpy array
        :param duration: The length of the timeseries, in seconds
        :type duration: float
        :return: the timeseries with added noise
        :rtype: numpy array
        """
        fs = 4096
        df = 1 / duration
        f_min = 10

        lo = int(f_min/df)
        N_fd = np.floor(fs * duration / 2.0 - lo)
        Nt_noise = int(fs * duration)

        # Resample PSD
        freq_values = np.arange(10, 2048, df)
        f = interpolate.interp1d(PSD[0], PSD[1])
        PSD = f(freq_values)
        
        # Now construct Gaussian data series
        Real = np.random.normal(0,1,size=int(N_fd))*np.sqrt(PSD/(4.*df))
        Imag = np.random.normal(0,1,size=int(N_fd))*np.sqrt(PSD/(4.*df))

        # Create data series as data = real + i*imag
        detData = (Real + 1j*Imag) 

        # Convert back into time domain
        time_domain_noise = np.fft.irfft(detData, n=Nt_noise) * fs   
        timeseries += time_domain_noise

        return timeseries

