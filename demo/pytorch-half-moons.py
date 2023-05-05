import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs

# for reproducibility
np.random.seed(42)
random.seed(42)

# make up a dataset
X, y = make_moons(n_samples=1000, noise=0.1)

y = y*2 - 1 # make y be -1 or 1

# convert data to PyTorch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float().unsqueeze(1)

# define the model
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(sizes)-1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(2, [16, 16], 1)
print(model)
print("number of parameters", sum(p.numel() for p in model.parameters()))

# define the loss function
def loss(X, y):
    scores = model(X)
    margin = y * scores
    losses = (1 - margin).clamp(min=0)
    data_loss = losses.mean()
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum(p.norm(2)**2 for p in model.parameters())
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = (scores.sign() == y.sign()).float().mean()
    return total_loss, accuracy

def visualize_decision_boundary(model, ax):
    # visualize decision boundary
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    Xmesh_torch = torch.from_numpy(Xmesh).float()
    scores = model(Xmesh_torch)
    Z = (scores > 0).float().reshape(xx.shape)

    # Update the data on the plot
    ax.clear()
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y.squeeze().numpy(), s=40, cmap=plt.cm.Spectral)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

# create a figure and axis object for the plot
fig, ax = plt.subplots(figsize=(5,5))

total_loss, acc = loss(X, y)
print(total_loss, acc)

# Show the initial plot
visualize_decision_boundary(model, ax)
plt.show(block=False)

# optimization
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
for k in range(100):
    # forward
    total_loss, acc = loss(X, y)

    # backward
    optimizer.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    # use PyTorch optimizer instead
    optimizer.step()

    if k % 1 == 0:
        print(f"step {k} loss {total_loss.item()}, accuracy {acc.item()*100}%")
        
    if k % 5 == 0:
        # Update the plot during each iteration
        visualize_decision_boundary(model, ax)
        plt.pause(0.001)

# Keep the plot window open until the user closes it manually
plt.show(block=True)