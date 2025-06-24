import torch
import torch.nn as nn
import torch.optim as optim

# 1. Prepare Data (y = 2x + 1)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 2. Define Model
model = nn.Linear(1, 1)

# 3. Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training Loop
for epoch in range(1000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# 5. Test the Model
with torch.no_grad():
    test = torch.tensor([[5.0]])
    prediction = model(test)
    print(f'Prediction for input 5: {prediction.item():.2f}')

