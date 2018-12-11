import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

batch_size = 5
lr = 1e-5

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')
                   
# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')
                    
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

train_ds = TensorDataset(inputs, targets)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
next(iter(train_dl))

# Define model
model = nn.Linear(3,2)

# SGD optimizer
opt = torch.optim.SGD(model.parameters(), lr)

# Loss function
loss_fn = F.mse_loss
loss = loss_fn(model(inputs),targets)

# Train the model
def fit(num_epochs, model, loss_fn, opt):
  for e in range(num_epochs):
    for x_batch, y_batch in train_dl:
      preds = model(x_batch)
      loss = loss_fn(preds, y_batch)
      print("Epoch {} and loss {}".format(e,loss))
      loss.backward()
      opt.step()
      opt.zero_grad()

fit(500, model, loss_fn, opt)
