import cv2
import numpy as np

# YOLOv4 모델과 가중치 파일 경로
model_path = "yolov4-fire_final.weights"
config_path = "yolov4-fire.cfg"

# YOLOv4 모델 로드
net = cv2.dnn.readNetFromDarknet(config_path, model_path)

# net = cv2.dnn.readNet(model_path, config_path)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# GPU를 사용할 경우
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 클래스 이름 파일 경로
classes_file = "fire.names"

# 클래스 이름 로드
with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 파일 경로
image_path = "samples/test3.jpg"


for i in range(1):


    # 이미지 로드
    image = cv2.imread(image_path)

    # 이미지 크기 가져오기
    height, width = image.shape[:2]

    # 이미지를 YOLOv4 모델 입력 크기에 맞게 리사이징
    scale = 1 / 255.0
    size = (416, 416)
    blob = cv2.dnn.blobFromImage(image, scale, size, swapRB=True, crop=False)

    # YOLOv4 모델에 이미지 입력
    net.setInput(blob)

    # YOLOv4 모델 실행
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # 감지된 객체를 저장할 리스트 초기화
    boxes = []
    confidences = []
    class_ids = []

    # 감지된 객체 정보 추출
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS(Normalized Maximum Suppression) 적용
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 객체 경계 상자 그리기
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        color = colors[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 결과 이미지 저장
    cv2.imwrite("result.jpg", image)

