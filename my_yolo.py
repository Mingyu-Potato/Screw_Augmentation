import torch
import numpy as np
import cv2
# import pafy
from time import time


def load_model():
    # YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def score_frame(model,frame):
        # frame: 단일 프레임; numpy/list/tuple 형식
        # return: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        frame = [frame]
        results = model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord
def class_to_label(model,x):
        # x 숫자 레이블 -> 문자열 레이블로 반환
        classes = model.names
        return classes[int(x)]
    
    
def plot_boxes(model,results, frame):
        # 경계상자와 레이블을 프레임에 플로팅
        # results: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        # frame: 점수화된 프레임
        # return: 경계 상자와 레이블이 플로팅된 프레임
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, class_to_label(model,labels[i])
                            + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 2)
        return frame
    
    
yolo_model = load_model()

img = cv2.imread('./송전철탑 주요 체결부 시편 사진/DSC01485.JPG')
img2 = img.copy()

result = yolo_model(img2)
img2 = cv2.resize(img2, (480, 256))
result = score_frame(yolo_model, img2)
result = plot_boxes(yolo_model, result, img2)

cv2.imshow('result', result)
cv2.waitKey(0)
if cv2.waitKey(1) & 0xFF == 27:
        cv2.release()
        cv2.destroyAllWindows()