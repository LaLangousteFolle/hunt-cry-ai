import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from src.dataset import HuntCryDataset
from src.model import HuntCryClassifier


def main():
    dataset = HuntCryDataset()

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HuntCryClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.3f}, "
              f"val_acc={correct/total:.2f}")

    torch.save(model.state_dict(), "models/hunt_cry_cnn.pt")


if __name__ == "__main__":
    main()
