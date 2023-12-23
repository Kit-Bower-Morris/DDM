import numpy as np
import librosa

def test(mels, mel_lens):

    



    mel_len = mel_lens[0].item()
    print(mel_len)
    mel = mels[0, :mel_len].detach().transpose(0, 1)
    mel = mel.cpu().numpy()
    import soundfile as sf
    S = librosa.feature.inverse.mel_to_stft(mel, sr=22050, n_fft=1024, fmax=8000)
    y = librosa.griffinlim(S)
    sf.write('test_mel2.wav', y, samplerate=22050)

#mel = np.load('D:/Chimera/Chopra/dataset/data/preprocessed_data/mel/Chopra-mel-01_chunk3_split0.npy')
mel = np.load('C:/Users/kitbm/Documents/Hexagram/data_format/FastSpeech2/data/mels/01-2_chunk2_split0.npy')
mel_len = mel.shape
print(mel_len)
#mel = mel.T
mel_len = mel.shape
print(mel_len)

def denormalize(S):
    return (np.clip(S, 0, 1) * 100) -100
def db_to_amp(x):
    return np.power(10.0, x * 0.05)
#denormalized = denormalize(mel)
#amp_mel = db_to_amp(denormalized)
import soundfile as sf
S = librosa.feature.inverse.mel_to_stft(mel, power=1, sr=22050, n_fft=1024, fmax=8000.0)
y = librosa.griffinlim(S, n_iter=32, hop_length=256, win_length=1024)
sf.write('test_mel10.wav', y, samplerate=22050)

'''

my_audio_as_np_array, my_sample_rate= librosa.load("D:/Chimera/Chopra/dataset/data/wavs/01_chunk3_split0.wav")
mel = np.load('D:/Chimera/Chopra/dataset/data/preprocessed_data/mel/Chopra-mel-01_chunk3_split0.npy')

# step2 - converting audio np array to spectrogram
spec = librosa.feature.melspectrogram(y=my_audio_as_np_array,
                                        sr=my_sample_rate, 
                                            n_fft=1024, 
                                            hop_length=256, 
                                            win_length=None, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0,
                                     n_mels=80)
mel = mel.T
print(mel.shape)
print(spec.shape)
print(np.setdiff1d(mel, spec))
# step3 converting mel-spectrogrma back to wav file
res = librosa.feature.inverse.mel_to_audio(spec, 
                                           sr=my_sample_rate, 
                                           n_fft=1024, 
                                           hop_length=256, 
                                           win_length=None, 
                                           window='hann', 
                                           center=True, 
                                           pad_mode='reflect', 
                                           power=2.0, 
                                           n_iter=32)

# step4 - save it as a wav file
import soundfile as sf
sf.write("test1.wav", res, samplerate=my_sample_rate)
'''