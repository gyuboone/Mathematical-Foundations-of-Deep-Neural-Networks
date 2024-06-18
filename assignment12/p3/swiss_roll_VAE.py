import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Step 0 : Define training configurations
"""

batch_size = 64
learning_rate = 5e-4
num_epochs = 4000
reg_coeff = 500
device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""
Step 1 : Define custom dataset 
"""

def make_swiss_roll(n_samples=2000, noise = 1.0, dimension = 2, a = 20, b = 5):
    """
    Generate 2D swiss roll dataset 
    """
    t = 2 * np.pi * np.sqrt(np.random.uniform(0.25,4,n_samples))
    
    X = 0.1 * t * np.cos(t)
    Y = 0.1 * t * np.sin(t)
    
    errors = 0.025 * np.random.multivariate_normal(np.zeros(2), np.eye(dimension), size = n_samples)
    X += errors[:, 0]
    Y += errors[:, 1]
    return np.stack((X, Y)).T

def show_data(data, title):
    """
    Plot the data distribution
    """
    sns.set(rc={'axes.facecolor': 'honeydew', 'figure.figsize': (5.0, 5.0)})
    plt.figure(figsize = (5, 5))
    plt.rc('text', usetex = False)
    plt.rc('font', family = 'serif')
    plt.rc('font', size = 10)
    
    g = sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, thresh=0.1, levels=1000, cmap="Greens")
    
    g.grid(False)
    plt.margins(0, 0)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.title(title)
    plt.show()


"""
Step 2 : Define custom dataset and dataloader. 
"""

class SwissRollDataset(Dataset) : 
    def __init__(self, data) : 
        super().__init__()
        self.data = torch.from_numpy(data)
    
    def __len__(self) : 
        return len(self.data)
    
    def __getitem__(self, idx) :
        return self.data[idx]

    
data = make_swiss_roll()
dataset = SwissRollDataset(data)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)


sns.set(rc={'axes.facecolor': 'honeydew', 'figure.figsize': (5.0, 5.0)})
plt.figure(figsize = (5, 5))
plt.rc('text', usetex = False)
plt.rc('font', family = 'serif')
plt.rc('font', size = 10)

g = sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, thresh=0.1, levels=1000, cmap="Greens")

g.grid(False)
plt.margins(0, 0)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.title('p_data')
plt.savefig('swiss_roll_true.png')
plt.show()


"""
Step 3 : Implement models
"""

class Encoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.layer1 = nn.Linear(2, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, 2)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x),0.2)
        x = F.tanh(self.layer2(x))
        x = self.layer3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, width = 64):
        super().__init__()
        self.layer1 = nn.Linear(1, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, 2)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x),0.2)
        x = F.tanh(self.layer2(x))
        x = self.layer3(x)

        return x
    


"""
Step 4 : Train models
""" 

E = Encoder().to(device)
D = Decoder().to(device)

optimizer = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr = learning_rate)







for epoch in range(num_epochs) : 
    for batch_idx, x in enumerate(loader) : 

        x = x.view(x.shape[0], -1).to(torch.float32).to(device)

        optimizer.zero_grad()

        param = E(x)

        z = torch.randn(x.shape[0]).to(device)*param[:,1].exp() + param[:,0]
        z = z.view(-1, 1)
        z = z.to(device)

        output = D(z)

        loss = (((output - x) ** 2).sum() * 150 + param[:,1].exp().sum() + (param[:,0]**2).sum() - param[:,1].sum()) / batch_size

        loss.backward()
        optimizer.step()
        



    
    # Visualize the intermediate result
    if (epoch+1) % (num_epochs // 5) == 0:
        print(epoch)
        with torch.no_grad():
            Z = torch.randn(2000,1).to(device)
            viz = D(Z)
            viz = viz.cpu().numpy()


            sns.set(rc={'axes.facecolor': 'honeydew', 'figure.figsize': (5.0, 5.0)})
            plt.figure(figsize = (5, 5))
            plt.rc('text', usetex = False)
            plt.rc('font', family = 'serif')
            plt.rc('font', size = 10)

            g = sns.kdeplot(x=viz[:, 0], y=viz[:, 1], fill=True, thresh=0.1, levels=1000, cmap="Greens")

            g.grid(False)
            plt.margins(0, 0)
            plt.xlim(-1.5,1.5)
            plt.ylim(-1.5,1.5)
            plt.title(f"Epoch : {epoch}")
            plt.show()
        

with torch.no_grad():
    Z = torch.randn(2000,1).to(device)
    viz = D(Z)
    viz = viz.cpu().numpy()

    sns.set(rc={'axes.facecolor': 'honeydew', 'figure.figsize': (5.0, 5.0)})
    plt.figure(figsize = (5, 5))
    plt.rc('text', usetex = False)
    plt.rc('font', family = 'serif')
    plt.rc('font', size = 10)

    g = sns.kdeplot(x=viz[:, 0], y=viz[:, 1], fill=True, thresh=0.1, levels=1000, cmap="Greens")

    g.grid(False)
    plt.margins(0, 0)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.title(f"Epoch : {epoch}")
    plt.show()