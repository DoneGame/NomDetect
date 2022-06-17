from nomeroff_net import pipeline
from paddleocr import PaddleOCR
import cv2
import time
import re


def zeros(a):
    a = list(a)

    if a[0] == '0':
        a[0] = 'o'
    if a[1] == 'o':
        a[1] = '0'
    if a[2] == 'o':
        a[2] = '0'
    if a[3] == 'o':
        a[3] = '0'
    if a[4] == '0':
        a[4] = 'o'
    if a[5] == '0':
        a[5] = 'o'

    return ''.join(a)


th = 3
cnt = 2

vpath = r'videos\20220609_181518.mp4'

number_plate_detection = pipeline("number_plate_localization",
                                  image_loader="opencv")
ocr = PaddleOCR(lang='en')  # need to run only once to load model into memory
# cap = cv2.VideoCapture(vpath)
cap = cv2.VideoCapture('rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4')
# cap = cv2.VideoCapture(0)

print(cap.isOpened())

t = []
prev_frame_time = 0
n = 0
while cap.isOpened():

    n -= 1
    if n > 0:
        continue

    ret, frame = cap.read()

    if not ret:
        continue

    image = frame

    if image is None:
        continue

    start_time = time.time()

    output = number_plate_detection.forward([image])
    num_points = number_plate_detection.postprocess(output)[0][0]

    num_imgs = []
    for num in num_points:
        x1 = int(num[0]) - th
        x2 = int(num[2]) + th
        y1 = int(num[1]) - th
        y2 = int(num[3]) + th
        num_imgs.append(image[y1:y2, x1:x2])

    #if not num_imgs:
    #    n = cnt
    #    continue

    result = []
    for img in num_imgs:
        res = ocr.ocr(img, det=False, cls=False)
        for line in res:
            result.append(line[0])

    res = []
    for i in range(len(result)):
        a = result[i].lower()
        a = ''.join(filter(str.isalnum, a))
        if len(a) < 6:
            continue
        a = zeros(a)
        num = re.findall(r'[a-z][0-9]{3}[a-z]{2}', a)
        res.extend(num)

    t.append(time.time() - start_time)

    screen = image

    for num in num_points:
        x1 = int(num[0]) - th
        x2 = int(num[2]) + th
        y1 = int(num[1]) - th
        y2 = int(num[3]) + th
        cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))

    cv2.putText(screen, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(screen, str(res), (7, 150), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('video', screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("--- %s seconds ---" % (sum(t) / len(t)))
