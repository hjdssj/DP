import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from opacus import PrivacyEngine

# ---------- Model ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------- Data ----------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transform),
    batch_size=256,
    shuffle=True
)

# ---------- Training setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

privacy_engine = PrivacyEngine(
    model,
    sample_rate=256 / len(train_loader.dataset),
    noise_multiplier=1.0,     # σ
    max_grad_norm=1.0         # C
)

privacy_engine.attach(optimizer)

# ---------- Train ----------
for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta=1e-5)
    print(f"Epoch {epoch}: ε = {epsilon:.2f}, δ = 1e-5")
