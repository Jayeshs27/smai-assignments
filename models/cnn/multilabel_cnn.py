import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MultiLabelCNN(nn.Module):
    def __init__(self, 
                 num_classes, 
                 num_labels,
                 dropout_rate=0.0, 
                 optimizer='adam', 
                 learning_rate=0.001):
        
        super(MultiLabelCNN, self).__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        self.dropout_conv = nn.Dropout(p=dropout_rate)  

        self.fc1 = nn.Linear(64 * 16 * 16, 256)  
        self.dropout_fc = nn.Dropout(p=dropout_rate)  

        self.fc2 = nn.Linear(256, num_classes * num_labels)  
        self.loss_function = nn.CrossEntropyLoss()  
        self.train_losses = []
        self.val_losses = []

        self.init_optimizer(optimizer=optimizer)

    def init_optimizer(self, optimizer):
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(2)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = x.view(-1, 64 * 16 * 16)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, val_loader, device, num_epochs=10):
        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                    
                outputs = self.forward(images)
                self.optimizer.zero_grad()
            
                t_loss = self.loss(outputs, labels)

                t_loss.backward()
                self.optimizer.step()

                running_loss += t_loss.item()
    
            train_loss = running_loss / len(train_loader)
            val_loss, _, _ = model.evaluate(val_loader, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def predict(self, image, device):
        image = image.to(device)
        outputs = self.forward(image)

        outputs = outputs.reshape(-1, 3, 11)
        _, predicted = torch.max(outputs, dim=2) 
        return predicted 

    def loss(self, y_pred, y_true):
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for i in range(3):
            start = i * 11
            end = (i + 1) * 11
            target_idx = torch.argmax(y_true[:, start:end], dim=1)
            loss += criterion(y_pred[:, start:end], target_idx)
        return loss

    def evaluate(self, data_loader, device):
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        hamming_accuracy_sum = 0.0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                    
                outputs = self.forward(images)
                t_loss = self.loss(outputs, labels)
                running_loss += t_loss.item()

                outputs = outputs.reshape(-1, 3, 11)
                labels = labels.reshape(-1, 3, 11)
                labels = torch.argmax(labels, dim=2)

                _, predicted = torch.max(outputs, dim=2) 
                correct += (predicted == labels).all(dim=1).sum().item()
                total += labels.size(0)
             
                hamming_accuracy_sum += torch.mean((predicted == labels).float()) 
                
        loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        hamming_accuracy = hamming_accuracy_sum / len(data_loader)

        return loss, accuracy, hamming_accuracy