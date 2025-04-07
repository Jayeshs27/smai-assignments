from common import * 

def character_decoder():
    decoder = {}
    for i in range(26):
        decoder[i] = chr(65 + i)

    for i in range(26, 52):
        decoder[i] = chr(97 + (i - 26))
    decoder[52] = ''
    return decoder

def character_encoder():
    encoder = {}
    for i in range(26):
        encoder[chr(65 + i)] =  i
        encoder[chr(97 + i)] =  i + 26
    return encoder

class OCRModel(nn.Module):
    def __init__(self, input_dim=65536, hidden_dim=256, num_classes=53, num_layers=2):
        super(OCRModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, 
                          hidden_dim, 
                          num_layers, 
                          batch_first=True, 
                          dropout=0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)  
        x = self.flatten(x)
        x = self.fc1(x)
        x = x.unsqueeze(1).repeat(1, 32, 1) 
        rnn_out, _ = self.rnn(x)
        rnn_out = self.layer_norm(rnn_out) 
        output = self.fc2(rnn_out)
        return output
    
    def evaluate(self, val_loader, device):
        self.eval()
        self.to(device)

        correct_chars = 0
        total_chars = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = self(images)  
                predictions = torch.argmax(outputs, dim=2)  

                for pred, label in zip(predictions, labels):
                    non_null_mask = label != 52
                    correct_chars += (pred[non_null_mask] == label[non_null_mask]).sum().item()
                    total_chars += non_null_mask.sum().item()  
        
        avg_correct_chars = correct_chars / total_chars
        return avg_correct_chars
    
    def train_model(self, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        class_weights = torch.ones(53)
        class_weights[52] = 0.1  # Reduce weight for null character class
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # suitable loss for classification
        
        self.to(device)
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)  
                loss = criterion(outputs.permute(0, 2, 1), labels)  
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            val_ancc = self.evaluate(val_loader, device)
            train_loss = train_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation ANCC: {val_ancc:.4f}')