
#import os
import os
import numpy as np
import IPython.display as ipd
import torchaudio
import scipy.io.wavfile as wav
import torch
import csv

audios = {
    "drum": ["audios/drum/drum.wav", "audios/drum/drum_office.wav", "audios/drum/drum_operahall.wav", "audios/drum/drum_reverbed.wav"],
    "flute": ["audios/flute/flute.wav", "audios/flute/flute_office.wav", "audios/flute/flute_operahall.wav", "audios/flute/flute_reverbed.wav"],
    "guitar": ["audios/guitar/guitar.wav", "audios/guitar/guitar_office.wav", "audios/guitar/guitar_operahall.wav", "audios/guitar/guitar_reverbed.wav"],
    "piano": ["audios/piano/piano.wav", "audios/piano/piano_office.wav", "audios/piano/piano_operahall.wav", "audios/piano/piano_reverbed.wav"],
    "trumpet": ["audios/trumpet/trumpet.wav", "audios/trumpet/trumpet_office.wav", "audios/trumpet/trumpet_operahall.wav", "audios/trumpet/trumpet_reverbed.wav"]
}
def load_audio_file(filepath):
    if os.path.exists(filepath):
        return torchaudio.load(filepath)
    else:
        print(f"File not found: {filepath}")
        return None, None 
def psycho_acoustic_pre_filter(audio, sr):
    pre_emphasis = 0.97
    filtered_audio = np.append(audio[0, 0], audio[0, 1:] - pre_emphasis * audio[0, :-1])
    return torch.tensor(filtered_audio).unsqueeze(0)

def perceptual_loss(audio1, audio2, sr):
    filtered_audio1 = psycho_acoustic_pre_filter(audio1, sr)
    filtered_audio2 = psycho_acoustic_pre_filter(audio2, sr)
    
    min_length = min(filtered_audio1.shape[1], filtered_audio2.shape[1])
    filtered_audio2 = filtered_audio2[:, :min_length]
    
    loss_mse = torch.nn.MSELoss()
    return loss_mse(filtered_audio1, filtered_audio2)

def match_audio_length(audio1, audio2):
    min_length = min(audio1.shape[1], audio2.shape[1])
    audio1 = audio1[:, :min_length]
    audio2 = audio2[:, :min_length]
    return audio1, audio2


def calculate_mse(audios):
    for instrument, files in audios.items():
        
        # Load original audio files
        audio_wav, sr_wav = load_audio_file(files[0])
        audio_office, sr_office = load_audio_file(files[1])
        audio_operahall, sr_operahall = load_audio_file(files[2])
        audio_reverbed, sr_reverbed = load_audio_file(files[3])
        
        # Calculate MSE Loss for actual audio
        audio_office, audio_wav = match_audio_length(audio_office, audio_wav)
        audio_operahall, audio_wav = match_audio_length(audio_operahall, audio_wav)
        audio_reverbed, audio_wav = match_audio_length(audio_reverbed, audio_wav)

        mse_office_original = torch.nn.MSELoss()(audio_office, audio_wav)
        #print(f'MSE Loss (office and original - {instrument}):', mse_office_original.item() * 100)

        mse_operahall_original = torch.nn.MSELoss()(audio_operahall, audio_wav)
        #print(f'MSE Loss (opera hall and original - {instrument}):', mse_operahall_original.item() * 100)

        mse_reverbed_original = torch.nn.MSELoss()(audio_reverbed, audio_wav)
        #print(f'MSE Loss (reverbed and original - {instrument}):', mse_reverbed_original.item() * 100)


def calculate_perceptual_loss(audios, output_file='perceptual_loss_only.csv'):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['instrument', 'condition', 'perceptual_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instrument, files in audios.items():
            # Load original audio files
            audio_wav, sr_wav = load_audio_file(files[0])
            audio_office, sr_office = load_audio_file(files[1])
            audio_operahall, sr_operahall = load_audio_file(files[2])
            audio_reverbed, sr_reverbed = load_audio_file(files[3])

            # Calculate perceptual loss for original vs. altered audio
            audio_wav, audio_wav = match_audio_length(audio_wav, audio_wav)
            audio_office, audio_wav = match_audio_length(audio_office, audio_wav)
            audio_operahall, audio_wav = match_audio_length(audio_operahall, audio_wav)
            audio_reverbed, audio_wav = match_audio_length(audio_reverbed, audio_wav)
            
            perceptual_losses = [
                 ('reference', perceptual_loss(audio_wav, audio_wav, sr_wav)),
                ('office', perceptual_loss(audio_office, audio_wav, sr_wav)),
                ('opera hall', perceptual_loss(audio_operahall, audio_wav, sr_wav)),
                ('reverbed', perceptual_loss(audio_reverbed, audio_wav, sr_wav))
            ]

            for condition, loss in perceptual_losses:
                writer.writerow({'instrument': instrument, 'condition': condition, 'perceptual_loss': loss.item() * 100})
                print(f'Perceptual Loss ({condition} and original - {instrument}):', loss.item() * 100)


calculate_mse(audios)
calculate_perceptual_loss(audios)