import essentia
import numpy as np
import matplotlib.pyplot as plt
import librosa
from utils import *
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredict2D
from matplotlib.animation import FuncAnimation, PillowWriter

def predict_valence_arousal_values(audio_file, save_to_json=False) -> np.ndarray:
    """_summary_

    Args:
        audio_file (path): path to audio file

    Returns:
        predictions (list): predictions of [valence, arousal] values with shape (n_samples, 2)
        time_stamps (list): time stamps corresponding to the predictions
    """
    y, sr = librosa.load(audio_file, sr=22050)
    embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    embeddings = embedding_model(y)

    model = TensorflowPredict2D(graphFilename="models/deam-msd-musicnn-2.pb", output="model/Identity")

    # predictions.shape
    # windowLength = 3 sec, hopLength = 1.5 seconds
    pred = model(embeddings)
    predictions = np.array(normalize(pred))
    time_stamps = np.linspace(0, get_duration(audio_file=audio_file), num=len(predictions), endpoint=False)
    if save_to_json:
        save_as_json('valence_arousal', predictions.tolist())
        save_as_json('time_stamps', time_stamps.tolist())

    return predictions, time_stamps



def soft_classify_emotion(emotion_centers, prediction, sigma=0.3):

    """classify a prediction in the valence-arousal space to different emotions using RBF kernel

    Args:
        emotion_centers (dict): dictionary of emotion centers in the valence-arousal space. For example,
            {
                "happy": [0.8, 0.8],
                "sad": [-0.8, -0.8],
                "angry": [-0.8, 0.8],
                "relaxed": [0.8, -0.8],
                "neutral": [0.0, 0.0]
            }
        point (list): a point in the valence-arousal space
        sigma (float, optional): bandwidth of the RBF kernel. Defaults to 0.3.

    Returns:
        weights: How much the point belongs to each emotion
    """
    #Uses Radial Basis Function Kernel
    weights = {}
    
    for emotion, center in emotion_centers.items():
        dist = np.linalg.norm(prediction - center)  # Euclidean distance or other distance metric
        weights[emotion] = np.exp(-dist**2 / (2 * sigma**2))

    #normalize
    total = sum(weights.values())
    for k in weights:
        weights[k] /= total
        
    return weights




def animate_valence_arousal(predictions):
    """_summary_

    Args:
        predictions (list): _description_

    Returns:
        _type_: _description_
    """
    valence = predictions[:, 0]
    arousal = predictions[:, 1]
    n = len(predictions)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlabel("Valence", loc='right', labelpad=15)
    ax.set_ylabel("Arousal", loc='top', labelpad=15)
    ax.set_title("VA on 2D plane")

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    dot, = ax.plot([],[], marker='o', markersize=8, linestyle="")

    def init():
        dot.set_data([],[])
        return dot,

    def update(frame):
        # X means the animation is ending
        if frame >= n-5:
            dot.set_marker('X')
        dot.set_data([valence[frame]], [arousal[frame]])
        return dot,
    
    animation = FuncAnimation(fig, update, frames=n, init_func=init, blit=True)
    animation.save('VA_on_2d_plane.gif', writer=PillowWriter(fps=8))
    plt.show()
    return


if __name__ == '__main__':
    # print(predictions)
    AUDIO = 'audio/testSong.mp3'
    valence = predict_valence_arousal_values(AUDIO)[:,0]
    arousal = predict_valence_arousal_values(AUDIO)[:,1]
    times = np.arange(len(valence)) * 1.5  # hop length is 1.5 seconds

    plt.figure(figsize=(10, 5))
    plt.plot(times, valence, label='Valence', color='b')
    plt.plot(times, arousal, label='Arousal', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title('Valence and Arousal over Time')
    plt.legend()
    plt.grid()
    plt.show()