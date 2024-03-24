import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

alpha = 0.1
K = 1000
B = 128
N = 512

def f_true(x) :
    return (x-2) * np.cos(x*4)

torch.manual_seed(0)
X_train = torch.normal(0.0, 1.0, (N,))

# some noise
# y_train = f_true(X_train)
y_train = f_true(X_train) + torch.normal(0, 0.5, X_train.shape)

X_val = torch.normal(0.0, 1.0, (N//5,))
y_val = f_true(X_val)

train_dataloader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=B, shuffle=True)
test_dataloader = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=B)


# unsqueeze(1) reshapes the data into dimension [N,1],
# where is 1 the dimension of an data point.

# The batchsize of the test dataloader should not affect the test result
# so setting batch_size=N may simplify your code.
# In practice, however, the batchsize for the training dataloader
# is usually chosen to be as large as possible while not exceeding
# the memory size of the GPU. In such cases, it is not possible to
# use a larger batchsize for the test dataloader.


class MLP(nn.Module):
    def __init__(self, input_dim = 1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64, bias=True)
        self.linear2 = nn.Linear(64,64, bias=True)
        self.linear3 = nn.Linear(64,1, bias=True)
        self.layers = [self.linear1, nn.Sigmoid(), self.linear2, nn.Sigmoid() ,self.linear3]
    def forward(self, x):
        x = nn.Sequential(*self.layers)(x)
        return x
    
model = MLP()

for i in range(0,len(model.layers),2):
    layer = model.layers[i]
    layer.weight.data = torch.normal(0, 1, layer.weight.shape)
    layer.bias.data = torch.full(layer.bias.shape, 0.03)

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = alpha)

total_parameter = 0
for parameter in model.parameters():
    print(parameter.shape)
    total_parameter+=np.product(parameter.shape)
print()
print('total:',total_parameter)

# training
for _ in range(K):
    for x, y in train_dataloader:
        optimizer.zero_grad()

        train_loss = loss_function(model(x), y)*0.5
        train_loss.backward()

        optimizer.step()

print(model.linear2.bias)

# plotting
with torch.no_grad():
    xx = torch.linspace(-2,2,1024).unsqueeze(1)
    plt.plot(X_train,y_train,'rx',label='Data points')
    plt.plot(xx,f_true(xx),'r',label='True Fn')
    plt.plot(xx, model(xx),label='Learned Fn')
plt.legend()
plt.savefig('p2.png')
plt.show()


# When plotting torch tensors, you want to work with the
# torch.no_grad() context manager.

# When you call plt.plot(...) the torch tensors are first converted into
# numpy arrays and then the plotting proceeds.
# However, our trainable model has requires_grad=True to allow automatic
# gradient computation via backprop, and this option prevents 
# converting the torch tensor output by the model to a numpy array.
# Using the torch.no_grad() context manager resolves this problem
# as all tensors are set to requires_grad=False within the context manager.

# An alternative to using the context manager is to do 
# plt.plot(xx, model(xx).detach().clone())
# The .detach().clone() operation create a copied pytorch tensor that
# has requires_grad=False.

# To be more precise, .detach() creates another tensor with requires_grad=False
# (it is detached from the computation graph) but this tensor shares the same
# underlying data with the original tensor. Therefore, this is not a genuine
# copy (not a deep copy) and modifying the detached tensor will affect the 
# original tensor is weird ways. The .clone() further proceeds to create a
# genuine copy of the detached tensor, and one can freely manipulate and change it.
# (For the purposes of plotting, it is fine to just call .detach() without
# .clone() since plotting does not change the tensor.)

# This discussion will likely not make sense to most students at this point of the course.
# We will revisit this issue after we cover backpropagation.
