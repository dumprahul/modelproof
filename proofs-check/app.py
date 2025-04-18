import torch
import torch.nn as nn
import torch.optim as optim

# 1. Simulated dataset: Celsius temps and hot/cold label
# Below 20°C = cold (0), 20°C and above = hot (1)
temps = torch.tensor([[t] for t in range(-10, 41)], dtype=torch.float32)  # -10°C to 40°C
labels = torch.tensor([0 if t < 20 else 1 for t in range(-10, 41)], dtype=torch.long)

# 2. Define a very small model
class TempClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 2)  # output: hot or cold
        )

    def forward(self, x):
        return self.net(x)

model = TempClassifier()

# 3. Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(temps)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. Test with user input
def predict_temp(celsius):
    with torch.no_grad():
        x = torch.tensor([[celsius]], dtype=torch.float32)
        output = model(x)
        predicted = torch.argmax(output, dim=1).item()
        return "Hot 🔥" if predicted == 1 else "Cold ❄️"

# Try it out
print(predict_temp(5))   # ❄️
print(predict_temp(25))  # 🔥

# 5. Export to ONNX
dummy_input = torch.tensor([[0.0]], dtype=torch.float32)
torch.onnx.export(model, dummy_input, "hot.onnx",opset_version=10)

print("✅ Exported is_hot.onnx")
