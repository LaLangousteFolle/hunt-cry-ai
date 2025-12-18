import torch
from src.model import HuntCryClassifier
from src.audio import audio_to_mel

IDX2CLASS = {0: "injured", 1: "kill", 2: "headshot"}

def predict_one(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HuntCryClassifier().to(device)
    state = torch.load("models/hunt_cry_cnn.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = audio_to_mel(path)          
    x = x.unsqueeze(0).to(device)   

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()

    probs = out.softmax(1)[0].cpu().numpy()
    print("probs:", dict(zip(IDX2CLASS.values(), probs)))
    print(f"{path} -> {IDX2CLASS[pred]}")

if __name__ == "__main__":
    file_path = input("Path to wav file: ")
    predict_one(file_path)
