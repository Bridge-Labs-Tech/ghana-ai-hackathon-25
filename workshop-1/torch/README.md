# Simple PyTorch Linear Regression Example

This guide demonstrates how to train a simple linear regression model using PyTorch.

---

## 1. Install PyTorch

If you haven't already, install PyTorch:

```bash
pip install torch
```

---

## 2. Example: Linear Regression with PyTorch

Create a file named `simple_linear_regression.py` with the following code:

```python
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
```

---

## 3. Step-by-Step Notes

1. **Prepare Data**:

   - We use a simple linear relationship: `y = 2x + 1`.
   - `X` and `y` are tensors containing our training data.

2. **Define Model**:

   - `nn.Linear(1, 1)` creates a linear model with one input and one output.

3. **Loss and Optimizer**:

   - `MSELoss` is Mean Squared Error, suitable for regression.
   - `SGD` is Stochastic Gradient Descent.

4. **Training Loop**:

   - For 1000 epochs, the model predicts outputs, computes loss, backpropagates, and updates weights.
   - Every 100 epochs, it prints the current loss.

5. **Test the Model**:
   - After training, we test the model with a new input (`x=5`) and print the prediction.

---

## 4. Run the Program

In your terminal, run:

```bash
python simple_linear_regression.py
```

You should see the loss decreasing and a prediction close to 11 for input 5 (since `y = 2*5 + 1 = 11`).

---

## 5. Summary

- **Install PyTorch**: `pip install torch`
- **Write the code** (see above)
- **Run the script**: `python simple_linear_regression.py`
- **Observe**: Loss decreases, and the model predicts values close to the true function.
