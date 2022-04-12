
import librosa 
import numpy as np
import os
import pickle

class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal



class Padder:
	#Responsible to apply Padding to an array
	def __init__(self, mode="constant"):
		# mode : constnt, max, min ,mean (hyperparemeter for padding)
		self.mode=mode

	def left_pad(self, array, num_missing_items):
		# [1 2 3] --->2  --> [0 0 1 2 3] ZERO LEFT PADDING
		padded_array=np.pad(array, (num_missing_items, 0), mode= self.mode)
		return padded_array

		pass
	def right_pad(self, num_missing_items):
		# [1 2 3] --->2  --> [1 2 3 0 0] ZERO RIGHT PADDING
		padded_array=np.pad(array, (0, num_missing_items), mode= self.mode)
		return padded_array

class LogSpectrogramExtractor:
	#FEATURE EXTRACION: LOG ESPECTROGRAM EXTRACTOR: Log Espectgograms (dB) from a time-series signal.
	def __init__(self, frame_size, hop_length):
		self.frame_size=frame_size
		self.hop_length=hop_length
	def extract(self, signal):
		stft = librosa.stft(signal, n_fft= sself.frame_size, hop_length= self.hop_length)[:-1]
		 # (1 + frame_size / 2 , num_frames)
		 # 1024 -> 513 -> 512 = [:-1]
		spectogram= np.abs(stft)
		log_spectogram = librosa.amplitud_to_db(spectogram)
		return log_spectogram

class MinMaxNormaliser:
	# Applies Min-Max Noralization to an array
	def __init__(self, min_val, max_val):
		self.min=min_val
		self.max=max_val

	def normalise(self, array):
		norm_array = (array - array.min()) / (array.max() - array.min())  # -> [0 , 1] NORMALIZING BETWEEN 0 1
		norm_array = norm_array * (self.max - self.min) + self.min
		return norm_array

	def denormalise(self, norm_array, origianl_min, original_max):
		array = (norm_array - self.min) / (self.max - self.min)
		array = array * (original_max - origianl_min) + origianl_min
		return array

class Saver:
	#Rsponsabile to save features and the min max values

	def __init__(self, feature_save_dir, min_max_values_save_dir):
		self.feature_save_dir = feature_save_dir
		self.min_max_values_save_dir = min_max_values_save_dir

	def save_feature(norm_feature, file_path):
		save_path = self._generate_save_path(file_path)
		np.save(save_path, feature)

	def _generate_save_path(self, file_path):
		file_name = os.path.split(file_path)[1]
		save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
		return save_path

	def save_min_max_values(self, min_max_values):
		save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
		self._save(min_max_values, save_path)

	@staticmethod
	def _save(self, sata, save_path):
		with open(save_path, "wb") as f:
			pickle.dump(data, f)

class PreprocessingPipeline:
	# Precesses audio files in a directory, applying the following stesp to eachs file
	""" 1- Load a file
		2- Pad the signal if necessary
		3- Extracting log spectogram from signal (usign lebrosa library)
		4- Normalise spectogram
		5- Save the normalized spectogram

		6- Storing the min, max values for all the log spectograms  """
	def __init__(self):
		self.loader = None
		self.padder = None
		self.extractor = None
		self.normalizer = None
		self.saver= None
		self.min_max_values = {}
		self._loader=None
		self._num_expected_samples= None



	@property
	def loader(self):
		return self._loader

	@loader.setter
	def loader(self, loader):
		self._loader = loader 
		self._num_expected_samples = int(loader.sample_rate *loader.duration )

	def process(self, audio_files_dir):
		for root, _, files in os.walk(audio_files_dir):
			for file in files:
				file_path=os.path.join(root, file)
				self._process_file(file_path)
				print(f"Processed file {file_path}")
		self.saver.save_min_max_values(self.min_max_values)

	def _process_file(self, file_path):
		signal =self.loader.load(file_path)
		if self._is_padding_necessary(signal):
			signal = self._apply_padding(signal)
		feature = self.extractor.extract(signal)
		norm_feature = self.normalizer.normalise(feature)
		save_path =self.saver.save_feature(norm_feature, file_path)
		self._store_min_max_value(save_path, feature.min(), feature.max())

	def _is_padding_necessary(self, signal):
		if len(signal) < self._num_expected_samples:
			return True
		return False
	def _apply_padding(self, signal):
		num_missing_samples = self._num_expected_samples - len (signal)
		padded_signal = self.padder.right_pad(signal, num_missing_samples)
		return padded_signal

	def _store_min_max_value(self, save_path, min_val, max_val):
		self.min_max_values[save_path] = {
		"min": min_val,
		"max" : max_val
		}


if __name__=="__main__":
	FRAME_SIZE = 512
	HOP_LENGTH = 256
	DURATION = 0.74 #in seconds 
	SAMPLE_RATE = 22050
	MONO= True

	SPECTOGRAMS_SAVE_DIR = "/home/kaliw/GITHUB/AUDIO/SpokenDigits/spectograms/"
	MIN_MAX_VALUES_SAVE_DIR = "/home/kaliw/GITHUB/AUDIO/SpokenDigits/min_max_values/"
	FILES_DIR = "/home/kaliw/GITHUB/AUDIO/SpokenDigits/DATA"


	#instantieate all objects

	loader = Loader(SAMPLE_RATE, DURATION, MONO)
	padder = Padder()
	log_spectogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
	min_max_normaliser = MinMaxNormaliser(0,1)
	saver = Saver(SPECTOGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

	Pipeline= PreprocessingPipeline()
	Pipeline.loader= loader
	Pipeline.padder= padder
	Pipeline.extractor= extractor
	Pipeline.normalizer= normalizer
	Pipeline.saver= saver

	Pipeline.process(FILES_DIR)
