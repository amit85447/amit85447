import torch
import torch.nn as nn
class Net(nn.Module):
  def __init__(self,hidden_size,dropout):
    super().__init__() # Fixed: changed super(),__init__() to super().__init__()
    self.fc1=nn.Linear(10,hidden_size)
    self.relu=nn.ReLU() # Fixed: changed nn.Relu() to nn.ReLU()
    self.dropout=nn.Dropout(dropout)
    self.fc2=nn.Linear(hidden_size,1)

  def forward(self,x):
    x=self.fc1(x)
    x=self.relu(x) # Fixed: changed X to x
    x=self.dropout(x)
    x=self.fc2(x)
    return x

def train_model(model, optimizer, criterion, x, y, epochs=50):
  model.train() # Set model to training mode
  for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
  return loss.item() # Return the last loss after training

learning_rates=[0.1,0.01,0.0001] # Fixed: changed 00.1 to 0.01
hidden_sizes=[16,32,64]
dropouts=[0.0,0.2,0.5]
best_loss=float('inf')

batch_size=64
x = torch.randn(batch_size,10)
y = torch.randn(batch_size,1)

best_params=None
best_model = None # Initialized best_model

for lr in learning_rates:
  for hs in hidden_sizes:
    for dp in dropouts:
      model=Net(hs,dp)
      optimizer=torch.optim.Adam(model.parameters(),lr=lr)
      criterion=nn.MSELoss()
      # Removed train_loader from arguments as it's not used in train_model
      loss=train_model(model,optimizer,criterion,x,y)
      if loss < best_loss: # Fixed: changed loss<loss to loss < best_loss
        best_loss=loss
        best_model=model
        best_params=(lr,hs,dp)

print(best_params)
print(best_loss)


optimizer=torch.optim.Adam(best_model.parameters(),lr=0.01)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)
for epoch in range(100):
  optimizer.zero_grad()
  output=best_model(x)
  loss=criterion(output,y)
  loss.backward()
  optimizer.step()
  scheduler.step()

import optuna

def objective(trail):
  lr=trail.suggest_loguniform('lr',1e-4,1e-1)
  hidden=trail.suggest_int('hidden',16,128)
  dropout=trail.suggest_uniform('dropout',0.0,0.5)
  model=Net(hidden,dropout)
  optimizer=torch.optim.Adam(model.parameters(),lr=lr)
  criterion=nn.MSELoss()
  loss=train_model(model,optimizer,criterion,x,y)
  return loss
  study=create_study(direction='minimize')
  study.optimize(objective,n_trials=50)
  print('study')

