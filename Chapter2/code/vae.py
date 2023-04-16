import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 1e-3

# Load the MNIST dataset
root = './data'
transform = transforms.Compose([transforms.ToTensor()])
train_set = dset.MNIST(root=root, train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

# Define the VAE model
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var




model = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    MSE = BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD



# Train the VAE
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(images)
        loss = loss_function(recon_batch, images, mu, logvar)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()/len(images)))


# Visualize the latent space
model.eval()

with torch.no_grad():
    # Prepare a batch of data for visualization
    data, targets = next(iter(train_loader))
    mu, _ = model.encoder(data.view(-1, 784))


    # Apply t-SNE on the latent space
    tsne = TSNE(n_components=2, random_state=42)
    mu_2D = tsne.fit_transform(mu.numpy())

    # Plot the 2D latent space with colors corresponding to digits
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mu_2D[:, 0], mu_2D[:, 1], c=targets, cmap='viridis', s=50)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of VAE Latent Space")
    plt.show()

