import numpy as np
import torch

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')
                    
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
learning_rate = 1e-5

weights = torch.randn(2,3, requires_grad=True)
bias = torch.randn(2, requires_grad=True)

def model(x):
  return x.mm(w.t()) + b
  
def mse(p,t):
  diff = p-t
  return torch.sum(diff*diff)/diff.numel()

for e in range(1500):
  preds = model(inputs)
  loss = mse(preds,targets)
  print("Epoch {} and loss {}".format(e,loss))
  loss.backward()
  with torch.no_grad():
    w -= w.grad*learning_rate
    b -= b.grad*learning_rate
    w.grad.zero_()
    b.grad.zero_()
    
preds = model(inputs)
print("Predicted values are {}".format(preds))
print("Target values were {}".format(targets))
