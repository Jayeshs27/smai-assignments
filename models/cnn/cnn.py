import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, 
                task='classification', 
                num_classes=10, 
                dropout_rate=0.0, 
                optimizer='adam', 
                learning_rate=0.001):
        
        super(CNN, self).__init__()
        self.task = task
        self.learing_rate = learning_rate
        self.num_classes = num_classes
        self.train_losses = []
        self.val_losses = []

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_conv = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(64 * 16 * 16, 256)  
        self.dropout_fc = nn.Dropout(p=dropout_rate)

        if self.task == 'classification':
            self.fc3 = nn.Linear(256, num_classes) 
            self.loss_function = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            self.fc3 = nn.Linear(256, 1) 
            self.loss_function = nn.MSELoss()

        self.init_optimizer(optimizer=optimizer)
    
    def init_optimizer(self, optimizer):
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learing_rate)
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learing_rate)
        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learing_rate)
        if optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.learing_rate)
        if optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.parameters(), lr=self.learing_rate)

    def forward(self, x):
        features_maps=[]

        if x.dim() == 5:
            x = x.squeeze(2) 

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        features_maps.append(x)
        x = self.dropout_conv(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        features_maps.append(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        features_maps.append(x)

        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        
        return x, features_maps

    def train_model(self, train_loader, val_loader, device, num_epochs=10):
        self.to(device)
        self.train() 
        self.train_losses=[]
        self.val_losses=[]

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                if self.task == 'regression':
                    labels = labels.float().unsqueeze(1)
                    
                outputs, _ = self.forward(images)
                loss = self.loss_function(outputs, labels)

                self.optimizer.zero_grad()  
                loss.backward() 
                self.optimizer.step() 

                running_loss += loss.item()
    
            train_loss = running_loss / len(train_loader)
            val_loss,_ = self.evaluate(val_loader, self.loss_function, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def predict(self, test_loader, device):
        self.to(device)
        self.eval()  
        predictions = []
        
        with torch.no_grad(): 
            for (images, labels) in test_loader:
                images = images.to(device)
                outputs, _ = self.forward(images)
                if self.task == 'classification':
                    outputs = F.log_softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)  
                elif self.task == 'regression':
                    predicted = outputs  
                predictions.append(predicted.cpu())
        
        return torch.cat(predictions, dim=0)


    def evaluate(self, val_loader, loss_function, device):
        self.to(device)
        self.eval()  
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): 
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if self.task == 'regression':
                    labels = labels.float().unsqueeze(1)
    
                outputs, _ = self.forward(images)
                loss = loss_function(outputs, labels)
                running_loss += loss.item()

                if self.task == 'classification':
                    outputs = F.log_softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    predicted = torch.round(outputs)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * correct / total if total > 0 else 0

        return avg_loss, accuracy