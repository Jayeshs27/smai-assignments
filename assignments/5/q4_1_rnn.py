from common import *
from bit_counting_model import BitCountingModel

class BinarySequencesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)
        label = torch.FloatTensor([self.labels[idx]])
        length = len(sequence)
        return sequence, label, length

## 4.1.1 Dataset Generation

def generate_binary_sequence_data(num_sequences=100000, min_length=1, max_length=16):
    sequences = []
    counts = []
    for _ in range(num_sequences):
        length = np.random.randint(min_length, max_length + 1)
        sequence = np.random.randint(0, 2, size=length).tolist()
        count_ones = sum(sequence)
        
        sequences.append(sequence)
        counts.append(count_ones)
    return sequences, counts

def preprocess_data(batch):
    sequences, labels, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    return sequences_padded, labels, lengths

def random_baseline(loader):
    total_loss = 0.0
    with torch.no_grad():
        for sequences, counts, _ in loader:
            counts = counts.numpy()
            random_predictions = np.random.randint(0, sequences.shape[-1] + 1, size=counts.shape)
            total_loss += np.sum(np.abs(random_predictions - counts))
    return total_loss / len(loader.dataset)


sequences, counts = generate_binary_sequence_data()

train_sequences, temp_sequences, train_counts, temp_counts = train_test_split(
    sequences, counts, test_size=0.2, random_state=42)
val_sequences, test_sequences, val_counts, test_counts = train_test_split(
    temp_sequences, temp_counts, test_size=0.5, random_state=42)

train_data = BinarySequencesDataset(train_sequences, train_counts)
val_data = BinarySequencesDataset(val_sequences, val_counts)
test_data = BinarySequencesDataset(test_sequences, test_counts)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=preprocess_data)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=preprocess_data)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=preprocess_data)


## 4.1.3 Model Training and Evaluation

model = BitCountingModel(input_size=1, hidden_size=16, output_size=1, num_layers=1, dropout=0.1).to(device)
model.train_model(train_loader, val_loader, num_epochs=10, learning_rate=0.001)

test_mae = model.evaluate_model(test_loader)
rnd_baseline_mae = random_baseline(test_loader)

print(f"Random Baseline MAE: {rnd_baseline_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")

## 4.1.4 Generalization on larger sequences

def generate_sequences_of_given_length(sequence_length, num_samples=1000):
    sequences = [np.random.randint(0, 2, sequence_length).tolist() for _ in range(num_samples)]
    counts = [sum(seq) for seq in sequences]
    return sequences, counts

def evaluate_data(model, data):
    model.eval()
    criterion = nn.L1Loss()  
    length_mae = {}
    with torch.no_grad():
        for length, dataset in data.items():
            loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=preprocess_data)
            total_loss = 0.0
            for sequences, counts, lengths in loader:
                sequences, counts, lengths = sequences.to(device), counts.to(device), lengths.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, counts)
                total_loss += loss.item()
            length_mae[length] = total_loss / len(loader)
    return length_mae

def plot_mae_vs_sequence_length():
    data = {}
    for length in range(1, 33):
        sequences, counts = generate_sequences_of_given_length(length)
        data[length] = BinarySequencesDataset(sequences, counts)

    MAEs = evaluate_data(model, data) 
    lengths, maes = zip(*sorted(MAEs.items()))

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, maes, marker='o', ms=3, color='g', label="MAE by Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("MAE vs. Sequence Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/rnn_seq_len_vs_mae.png')
    plt.show()

plot_mae_vs_sequence_length()



