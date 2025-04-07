from common import *

## 4.1.2 

class BitCountingModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1, dropout=0.0):
        super(BitCountingModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, lengths):
        rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(rnn_input)
        output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        output = output[torch.arange(output.size(0)), lengths - 1]
        output = self.fc(output)
        return output

    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.L1Loss()  
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for sequences, counts, lengths in train_loader:
                sequences, counts, lengths = sequences.to(device), counts.to(device), lengths.to(device)
                outputs = self(sequences, lengths)
                loss = criterion(outputs, counts)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() 

            val_loss = self.evaluate_model(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training MAE: {running_loss/len(train_loader):.4f}, Validation MAE: {val_loss:.4f}")

    def evaluate_model(self, loader):
        self.eval()
        criterion = nn.L1Loss()  
        total_loss = 0.0
        with torch.no_grad():
            for sequences, counts, lengths in loader:
                sequences, counts, lengths = sequences.to(device), counts.to(device), lengths.to(device)
                outputs = self(sequences, lengths)
                loss = criterion(outputs, counts)
                total_loss += loss.item() 
        return total_loss / len(loader)