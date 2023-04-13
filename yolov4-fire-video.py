import cv2
import numpy as np
# import DeviceManager

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

# 웹캠으로부터 입력 받기 위해 VideoCapture 객체 생성
cap = cv2.VideoCapture(0)
# 웹캠 프레임 사이즈 설정
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("YOLOv4-Fire", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv4-Fire", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:
    # 웹캠에서 프레임 읽어오기
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (416, 416))

    # 이미지 크기 가져오기
    height, width = frame.shape[:2]

    # 이미지를 YOLOv4 모델 입력 크기에 맞게 리사이징
    scale = 1 / 255.0
    size = (416, 416)
    blob = cv2.dnn.blobFromImage(frame, scale, size, swapRB=True, crop=False)

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
            if confidence > 0.2:
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
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.5)

    fireAreaClor = (0, 0, 255)
    if len(indexes) >= 1:
        # DeviceManager.fanStart()
        print(indexes)
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            color = (0, 255, 128)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if( (0 < (x+w/2) <= 208) and (0 < (y+h/2) <= 208) ):
                cv2.rectangle(frame, (0, 0), (208, 208), fireAreaClor, 2)
                cv2.putText(frame, "4 quadrant fire detection", (0, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            elif ( (208 < (x+w/2) <= 416) and (0 < (y+h/2) <= 208) ):
                cv2.rectangle(frame, (208, 0), (416, 208), fireAreaClor, 2)
                cv2.putText(frame, "1 quadrant fire detection", (0, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            elif ( (208 < (x+w/2) <= 416) and (208 < (y+h/2) <= 416) ):
                cv2.rectangle(frame, (208, 208), (416, 416), fireAreaClor, 2)
                cv2.putText(frame, "2 quadrant fire detection", (0, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            elif ( (0 < (x+w/2) <= 208) and (208 < (y+h/2) <= 416) ):
                cv2.rectangle(frame, (0, 208), (216, 416), fireAreaClor, 2)
                cv2.putText(frame, "3 quadrant fire detection", (0, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            else:
                print('')


            text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        # DeviceManager.fanStop()
        pass

    # 결과 이미지 출력
    frame = cv2.resize(frame, (1024, 600))
    cv2.imshow("YOLOv4-Fire", frame)
    if cv2.waitKey(1) == ord("q"):
        break

