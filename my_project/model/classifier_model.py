import os
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet  # pip install efficientnet_pytorch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


class EfficientClassifierModel(nn.Module):
    def __init__(self, num_classes=2, model_name="efficientnet-b0"):
        """
        EfficientNet 기반 분류 모델.
        model_name: efficientnet-pytorch의 이름 (예: 'efficientnet-b0')
        num_classes: 분류할 클래스 수 (예제에서는 2)
        """
        super(EfficientClassifierModel, self).__init__()
        # EfficientNet 구조 생성 (가중치는 나중에 로드)
        self.base_model = EfficientNet.from_name(model_name)
        in_features = self.base_model._fc.in_features
        # 마지막 fc 레이어를 원하는 num_classes에 맞게 재정의
        self.base_model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


class EfficientClassifier:
    def __init__(
        self,
        model_path="model/efficient_classifier.pth",
        num_classes=3,
        device=None,
        model_name="efficientnet-b0",
    ):
        """
        model_path: efficient_classifier.pth 파일 경로 (현재 작업 디렉토리 기준 model 폴더 내)
        num_classes: 분류할 클래스 수
        device: 모델이 배치될 장치 (없으면 CPU 사용)
        model_name: efficientnet_pytorch에서 사용할 모델 이름
        """
        self.device = device or torch.device("cpu")
        self.model = EfficientClassifierModel(
            num_classes=num_classes, model_name=model_name
        ).to(self.device)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            new_state_dict = {}
            # checkpoint가 원래 fc 레이어(여기서는 _fc)까지 포함되어 있으므로, fc 관련 키는 건너뛰고,
            # 나머지 키에 "base_model." 접두어를 붙여서 현재 모델의 상태 구조에 맞춥니다.
            for key, value in state_dict.items():
                # fc 레이어 관련 파라미터는 건너뜁니다.
                if key.startswith("_fc"):
                    continue
                # 만약 key에 이미 "base_model." 접두어가 없다면 붙여줍니다.
                new_key = key if key.startswith("base_model.") else "base_model." + key
                new_state_dict[new_key] = value

            # strict=False로 로딩하여 fc 관련 미스매치는 무시합니다.
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            print("EfficientClassifier 모델이 성공적으로 로드되었습니다.")
        else:
            print(f"{model_path} 파일이 존재하지 않습니다. 모델 파일을 확인하세요.")
            self.model = None

        # 전처리 transform (EfficientNet-B0 기준 ImageNet normalization)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # EfficientNet-B0의 일반 입력 크기
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, input_image):
        """
        input_image: cv2.imread()로 읽은 BGR 이미지 (numpy 배열)
        반환: 예측 클래스 인덱스와 softmax 확률 리스트
        """
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None

        # BGR → RGB 변환 및 전처리
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_idx = torch.max(probabilities, dim=1)

        return predicted_idx.item(), probabilities.squeeze().cpu().tolist()


def main():
    # device 설정
    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS를 사용합니다:", device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA GPU를 사용합니다:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CPU를 사용합니다.")

    # EfficientClassifier 인스턴스 생성 시 device 전달
    classifier = EfficientClassifier(
        model_path="model/efficient_classifier.pth",
        num_classes=2,
        model_name="efficientnet-b0",
        device=device,
    )
    if classifier.model is None:
        return

    # 이하 코드는 동일
    image_path = "resources/test/test1.png"
    if os.path.exists(image_path):
        input_image = cv2.imread(image_path)
    else:
        print("테스트 이미지가 존재하지 않습니다. 대신 dummy 이미지를 사용합니다.")
        input_image = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)

    predicted_class, confs = classifier.predict(input_image)
    print(f"예측 결과 클래스: {predicted_class}")
    print(f"확률: {confs}")

    annotated_image = input_image.copy()
    cv2.putText(
        annotated_image,
        f"Class: {predicted_class}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("EfficientClassifier Prediction", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
