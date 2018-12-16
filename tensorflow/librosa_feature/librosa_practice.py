import librosa
import librosa.display
import matplotlib.pyplot as plt

# load wav file (make sure u have happy.wav in ur dir)
# get sequence data and sample rate
y, sr = librosa.load('data/happy.wav', sr=None)

# extract mfcc feature
mfcc = librosa.feature.mfcc(y, sr, n_mfcc=40)
print(mfcc.shape)

# plot the wavform
plt.figure('happy.wav')
plt.subplot(2, 1, 1)
librosa.display.waveplot(y, sr)
plt.title('happy.wav')

# plot the mfcc
plt.subplot(2, 1, 2)
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('mfcc feature')
plt.tight_layout()  # show normally
plt.show()
