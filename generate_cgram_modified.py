import matplotlib
matplotlib.use('agg')


import librosa, librosa.display
from os import walk
from os.path import join, exists
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


INPUT_DIR = 'training-2022/training-2022'
OUTPUT_DIR = 'chroma-2022'


def generate_cgram(wav_file, sr, filename):
    cgram = librosa.feature.chroma_stft(y=wav_file, sr=sr, hop_length=512, n_fft=2048)
    plt.figure(figsize=(15,5)) 
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(cgram, sr=sr, hop_length=512, cmap='coolwarm')
    #plt.colorbar()
    plt.savefig(filename, bbox_inches=None, pad_inches=0)
    
    plt.close()


def get_output_name(filename, output_dir=OUTPUT_DIR):
    filename = filename.split('.')[0] + '.jpg'
    return(output_dir + '/' +filename)

if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for path, dirs, files in walk(INPUT_DIR):
        for filename in tqdm(files):
            if filename.endswith('.wav'):
                output_name = get_output_name(filename)
                if(exists(output_name)):
                    continue
                wav_file, sr = librosa.load(join(path, filename), sr=None, mono=True)
                generate_cgram(wav_file, sr, output_name)
