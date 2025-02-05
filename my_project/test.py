import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO  # pip install ultralytics 필요
from torchvision import transforms, models
import torch.nn as nn


##########################################
# MobileNetV3Classifier 관련 클래스 정의
##########################################
class MobileNetV3ClassifierModel(nn.Module):
    def __init__(self, num_classes=3, model_name="mobilenet_v3_small"):
        """
        MobileNetV3 기반 분류 모델.
        model_name: 'mobilenet_v3_small' 또는 'mobilenet_v3_large'
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
        반환: (예측 클래스 인덱스, softmax 확률 리스트)
        """
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None
        # BGR → RGB 변환 후, 전처리 파이프라인 실행 (RedRegionExtract 포함)
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_idx = torch.max(probabilities, dim=1)
        return predicted_idx.item(), probabilities.squeeze().cpu().tolist()


##########################################
# YOLO 모델 추론 및 통합 테스트 main() 함수
##########################################
def main():
    # 테스트 이미지 경로 (적절한 경로로 수정)
    test_image_path = "resources/test/test4.png"
    if not os.path.exists(test_image_path):
        print("테스트 이미지 파일을 찾을 수 없습니다:", test_image_path)
        return

    # OpenCV로 이미지 읽기 (BGR)
    image_bgr = cv2.imread(test_image_path)
    if image_bgr is None:
        print("이미지 로드 실패!")
        return

    # YOLO 모델 로드 (ultralytics 사용; 모델 파일 경로는 환경에 맞게 수정)
    yolo_model_path = os.path.join("model", "yolo11n.pt")
    if not os.path.exists(yolo_model_path):
        print("YOLO 모델 파일을 찾을 수 없습니다:", yolo_model_path)
        return
    yolo_model = YOLO(yolo_model_path)  # ultralytics.YOLO 클래스 사용

    # YOLO 추론 수행 (verbose=False로 불필요한 출력 제거)
    results = yolo_model(image_bgr, verbose=False)
    if not results or len(results) == 0:
        print("YOLO 추론 결과가 없습니다.")
        return

    # MobileNetV3Classifier 인스턴스 생성
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = MobileNetV3Classifier(
        model_path="model/best_mobileNetv3.pth",
        num_classes=3,
        model_name="mobilenet_v3_small",
        device=device,
    )

    # YOLO 결과에서 bounding box 추출 (yolo 모델 버전에 따라 구조 차이가 있을 수 있음)
    boxes = []
    for result in results:
        if result.boxes is not None:
            try:
                b = result.boxes.xyxy.cpu().numpy()
            except Exception:
                b = result.boxes.xyxy.numpy()
            boxes.extend(b.tolist())
    if not boxes:
        print("검출된 바운딩 박스가 없습니다.")
        return

    # 원본 이미지 복사 (BGR) -> annotated 이미지로 사용
    annotated_img = image_bgr.copy()

    # 각 bounding box에 대해:
    # 1. crop 이미지 추출 (YOLO crop)
    # 2. RedRegionExtract 전처리 결과 확인 (inside MobileNetV3Classifier의 transform에도 포함됨)
    # 3. MobileNetV3Classifier.predict() 통해 분류 수행
    # 각 결과를 annotated 이미지에 오버레이
    label_map = {0: "NG1", 1: "NG2", 2: "GOOD"}
    red_extractor = RedRegionExtract()

    # 각 검출된 물체 영역 별로 별도의 figure를 구성하여 결과 출력
    num_detections = len(boxes)
    fig, axes = plt.subplots(num_detections, 3, figsize=(15, 5 * num_detections))
    if num_detections == 1:
        axes = [axes]  # 1개일 경우 2차원 리스트로 변환

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        # YOLO crop 이미지 (원본 crop)
        crop_img = image_bgr[y1:y2, x1:x2].copy()
        # 전처리 결과: RedRegionExtract 적용 (결과는 PIL 이미지)
        crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        red_processed = red_extractor(crop_pil)

        # MobileNetV3Classifier.predict() 호출 (입력은 BGR 이미지; 내부 transform에 RedRegionExtract 포함됨)
        predicted, probabilities = classifier.predict(crop_img)
        prediction_label = label_map.get(predicted, str(predicted))

        # annotated 이미지에 bounding box와 예측 텍스트 오버레이
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_img,
            f"{prediction_label}",
            (x1, y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # 결과 서브플롯 출력
        # 서브플롯 1: YOLO crop 이미지 (원본)
        axes[i][0].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"YOLO Crop Region {i+1}")
        axes[i][0].axis("off")
        # 서브플롯 2: 전처리 (RedRegionExtract) 결과 이미지
        axes[i][1].imshow(red_processed)
        axes[i][1].set_title(f"Preprocessed (RedRegionExtract) {i+1}")
        axes[i][1].axis("off")
        # 서브플롯 3: 예측 결과 (텍스트 오버레이된 원본 crop 이미지)
        annotated_crop = crop_img.copy()
        cv2.putText(
            annotated_crop,
            f"{prediction_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        axes[i][2].imshow(cv2.cvtColor(annotated_crop, cv2.COLOR_BGR2RGB))
        axes[i][2].set_title(f"Classification: {prediction_label}")
        axes[i][2].axis("off")

    # 전체 annotated 이미지 (원본 이미지에 박스, 텍스트 오버레이)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title("Annotated Image (YOLO + MobileNetV3)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
