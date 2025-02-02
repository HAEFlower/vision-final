import os
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(26 * 26 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(-1, 26 * 26 * 16)
        x = self.fc1(x)
        return x


def load_model():
    model = SimpleModel()
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    else:
        print("model.pth가 존재하지 않습니다. 기본 랜덤 가중치를 사용합니다.")
    model.eval()
    return model


if __name__ == "__main__":
    # model.pth 파일 존재 여부에 따라서 처리
    loaded_model = load_model()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = loaded_model(dummy_input)
    print("추론 결과:", output)
