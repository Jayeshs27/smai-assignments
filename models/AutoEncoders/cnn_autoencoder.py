import torch
import torch.nn as nn


class CnnAutoencoder(nn.Module):
    def __init__(self,
                 learning_rate=0.001,
                 num_epochs=20,
                 optimizer='adam',
                 reduced_dim=128,
                 ):
        super(CnnAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, reduced_dim, kernel_size=7), 
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(reduced_dim, 64, kernel_size=7),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid() 
        )
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_function = nn.MSELoss()
        self.init_optimizer(optimizer=optimizer)

        self.train_losses=[]
        self.val_losses=[]

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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        latent_space = self.encode(x)  
        reconstructed = self.decode(latent_space)  
        return reconstructed

    def train_model(self, train_loader, val_loader, device):
        self.to(device) 
        self.train()
        self.train_losses=[]
        self.val_losses=[]

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for data in train_loader:
                img, _ = data 
                img = img.to(device)

                output = self.forward(img)
                loss = self.loss_function(output, img) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            val_loss = self.evaluate(val_loader, device)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def evaluate(self, val_loader, device):
        self.to(device)
        self.eval()  
        running_loss = 0.0

        with torch.no_grad(): 
            for data in val_loader:
                images, _ = data
                images = images.to(device)

                output = self.forward(images)
                loss = self.loss_function(output, images)
                running_loss += loss.item()

        loss = running_loss / len(val_loader)
        return loss