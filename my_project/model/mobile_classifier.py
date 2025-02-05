import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


##########################################
# MobileNetV3Classifier 관련 클래스 정의
##########################################
class MobileNetV3ClassifierModel(nn.Module):
    def __init__(self, num_classes=3, model_name="mobilenet_v3_small"):
        """
        MobileNetV3 기반 분류 모델.
        model_name: 'mobilenet_v3_small' 또는 'mobilenet_v3_large' (기본은 small)
        num_classes: 분류할 클래스 수
        """
        super(MobileNetV3ClassifierModel, self).__init__()
        if model_name == "mobilenet_v3_small":
            self.base_model = models.mobilenet_v3_small(pretrained=True)
        else:
            self.base_model = models.mobilenet_v3_large(pretrained=True)
        in_features = self.base_model.classifier[0].in_features
        # 기본 classifier를 num_classes에 맞는 단일 선형 계층으로 교체
        self.base_model.classifier = nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.base_model(x)


# 사용자 정의 전처리: 빨간색 영역 추출 및 정사각형 크롭
class RedRegionExtract(object):
    def __call__(self, image):
        """
        image: PIL 이미지 (RGB)
        반환: 빨간색 영역을 추출하여 정사각형 크롭한 PIL 이미지.
              빨간 영역을 찾지 못하면 원본 이미지를 그대로 반환.
        """
        print("RedRegionExtract 호출")
        # PIL(RGB) -> OpenCV(BGR): 마지막 축을 뒤집습니다.
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
            print("빨간색 영역을 찾을 수 없습니다. 원본 이미지를 그대로 사용합니다.")
            square_crop = image_cv

        # 다시 OpenCV(BGR) -> PIL(RGB)
        square_crop = cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(square_crop)


class MobileNetV3Classifier:
    def __init__(
        self,
        model_path="model/best_mobileNetv3.pth",
        num_classes=3,
        device=None,
        model_name="mobilenet_v3_small",
    ):
        """
        model_path: 저장된 모델 파일 경로 (모델 파라미터만 저장됨)
        num_classes: 분류할 클래스 수
        device: 모델이 배치될 장치 (없으면 CPU)
        model_name: 사용할 MobileNetV3 종류
        """
        self.device = device or torch.device("cpu")
        self.model = MobileNetV3ClassifierModel(
            num_classes=num_classes, model_name=model_name
        ).to(self.device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # fc층 미스매치는 무시
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print("MobileNetV3Classifier 모델이 성공적으로 로드되었습니다.")
        else:
            print(f"{model_path} 파일이 존재하지 않습니다. 모델 파일을 확인하세요.")
            self.model = None

        # 전처리 파이프라인: 빨간색 영역 추출(정사각형 크롭) → Resize → ToTensor → 정규화
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
        반환: (예측 클래스 인덱스, softmax 확률 리스트)
        """
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None

        # BGR → RGB 변환 후, transform 파이프라인 실행
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_idx = torch.max(probabilities, dim=1)
        return predicted_idx.item(), probabilities.squeeze().cpu().tolist()


def main():
    # 테스트 이미지 경로 (적절한 경로로 수정하세요)
    test_image_path = "resources/test/test2.png"
    if not os.path.exists(test_image_path):
        print("테스트 이미지 파일을 찾을 수 없습니다:", test_image_path)
        return

    # OpenCV로 이미지 읽기 (BGR 형식)
    image_bgr = cv2.imread(test_image_path)
    if image_bgr is None:
        print("이미지 로드 실패!")
        return

    # 장치 설정 (CUDA 사용 가능 시 GPU 사용)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MobileNetV3Classifier 인스턴스 생성
    classifier = MobileNetV3Classifier(
        model_path="model/best_mobileNetv3.pth",
        num_classes=3,
        model_name="mobilenet_v3_small",
        device=device,
    )

    # predict() 함수 호출 (입력은 BGR 형식의 numpy 배열)
    predicted, probabilities = classifier.predict(image_bgr)

    # 클래스 매핑: 0: NG1, 1: NG2, 2: GOOD
    label_map = {0: "NG1", 1: "NG2", 2: "GOOD"}
    prediction_label = label_map.get(predicted, str(predicted))

    # 예측 결과 출력
    print("Prediction:", prediction_label)
    print("Probabilities:", probabilities)

    # 원본 이미지를 RGB로 변환하여 화면에 표시
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f"Prediction: {prediction_label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
