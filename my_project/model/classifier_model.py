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


class RedRegionExtract(object):
    def __call__(self, image):
        """
        image: PIL 이미지 (RGB)
        반환: 빨간색 영역을 추출하여 정사각형 크롭한 PIL 이미지.
              빨간 영역을 찾지 못하면 원본 이미지를 그대로 반환.
        """
        print("RedRegionExtract 호출")
        # PIL(RGB) -> OpenCV(BGR): 마지막 축 뒤집기
        image_cv = np.array(image)[:, :, ::-1]
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        red_highlighted = cv2.bitwise_and(image_cv, image_cv, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            side_length = max(w, h)
            center_x, center_y = x + w // 2, y + h // 2
            half_side = side_length // 2
            start_x = max(center_x - half_side, 0)
            start_y = max(center_y - half_side, 0)
            end_x = min(center_x + half_side, image_cv.shape[1])
            end_y = min(center_y + half_side, image_cv.shape[0])
            square_crop = red_highlighted[start_y:end_y, start_x:end_x]
        else:
            print("빨간 영역을 찾을 수 없습니다. 원본 이미지를 그대로 사용합니다.")
            square_crop = image_cv
        # 다시 OpenCV(BGR) -> PIL(RGB)
        square_crop = cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(square_crop)


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

        # 전처리 파이프라인: RedRegionExtract → Resize → ToTensor → Normalize
        self.transform = transforms.Compose(
            [
                RedRegionExtract(),
                transforms.Resize((224, 224)),
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
