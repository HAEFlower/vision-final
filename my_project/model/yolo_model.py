import os
import torch
import cv2
from ultralytics import YOLO


def load_model():
    # 모델 파일은 현재 작업 디렉토리 기준으로 model 폴더 내에 있다고 가정합니다.
    model_path = os.path.join("model", "yolo11n.pt")
    if os.path.exists(model_path):
        # ultralytics의 YOLO 클래스는 사전 학습된 모델 파일을 로드합니다.
        model = YOLO(model_path)
    else:
        print(
            "yolo11n.pt 파일이 존재하지 않습니다. 사전 학습된 모델 파일을 확인해주세요."
        )
        model = None
    return model


def run_inference(yolo_model, input_image):
    """
    입력 이미지에 대해 inference를 수행하며, 'bottle' 클래스만 검출하도록 합니다.
    모델 내부의 names 딕셔너리에서 'bottle' 클래스에 해당하는 인덱스를 찾은 후,
    inference 시 classes 인자로 전달합니다.
    """
    target_class_idx = None
    if hasattr(yolo_model.model, "names"):
        for idx, name in yolo_model.model.names.items():
            if name.lower() == "bottle":
                target_class_idx = idx
                break

    if target_class_idx is None:
        print(
            "모델에서 'bottle' 클래스를 찾지 못했습니다. 기본적으로 클래스 0을 사용합니다."
        )
        target_class_idx = 0

    # inference 시 target_class_idx만 필터링하여 실행합니다.
    results = yolo_model(input_image, classes=[target_class_idx], verbose=False)
    return results


def main():
    yolo_model = load_model()
    if yolo_model is None:
        return

    # 테스트 이미지 경로 설정 (실제 bottle 이미지 경로로 변경하세요)
    image_path = "path/to/your/bottle_image.jpg"
    if os.path.exists(image_path):
        input_image = cv2.imread(image_path)
    else:
        print("테스트 이미지가 존재하지 않습니다. 대신 dummy 데이터를 사용합니다.")
        # dummy 데이터: (3 채널, 640x640) 랜덤 이미지 생성
        dummy_tensor = torch.randn(1, 3, 640, 640)
        # tensor를 numpy 배열 (H, W, C)로 변환 후 값 범위를 0~255로 정규화
        dummy_image = dummy_tensor.squeeze(0).permute(1, 2, 0).numpy()
        input_image = cv2.normalize(dummy_image, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )

    # inference 실행 (bottle 클래스만 검출)
    results = run_inference(yolo_model, input_image)

    # inference 결과가 있을 경우, 결과 이미지(바운딩 박스 등)가 오버레이된 annotated image 반환
    if results and len(results) > 0:
        # results[0].plot()는 검출 결과를 원본 이미지에 그린 annotated 이미지를 반환합니다.
        annotated_img = results[0].plot()
        cv2.imshow("Bottle Detections", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("추론 결과가 없습니다.")


if __name__ == "__main__":
    main()
