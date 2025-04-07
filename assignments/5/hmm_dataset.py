from common import *

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)  
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc) 
    return mfcc

def visualize_mfcc(mfcc_features, digit):
    plt.figure(figsize=(8, 5))
    sns.heatmap(mfcc_features, cmap='viridis')
    plt.title(f'MFCC Features Digit-{digit}')
    plt.xlabel('Time Frames')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.savefig(f"figures/digit_{digit}_mfcc_heatmap.png")
    # plt.show()

def generate_mfcc_features():
    data_dir = "../../data/external/spoken_digit_dataset/recordings"
    n_mfcc = 13
    digit_features = {i: [] for i in range(10)}
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            digit = int(filename.split("_")[0])
            file_path = os.path.join(data_dir, filename)
            mfcc_features = extract_mfcc(file_path, n_mfcc)
            digit_features[digit].append(mfcc_features)
    for digit in range(10):
        if digit_features[digit]:
            sample_features = digit_features[digit][0]
            visualize_mfcc(sample_features, digit)

if __name__ == "__main__":
    generate_mfcc_features()