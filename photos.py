from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from paddleocr import PaddleOCR
import cv2
import time
import re
import os


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

direc = r'photos'

number_plate_detection = pipeline("number_plate_localization",
                                  image_loader="opencv")
ocr = PaddleOCR(lang='en')  # need to run only once to load model into memory

ress = []
impaths = []
t = []
for impath in os.listdir(direc):

    image = cv2.imread(direc + '/' + impath)

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

    screen = image.copy()

    for num in num_points:
        x1 = int(num[0]) - th
        x2 = int(num[2]) + th
        y1 = int(num[1]) - th
        y2 = int(num[3]) + th
        cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(screen, str(res), (7, 70), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
    screen = cv2.resize(screen, (1920, 1080))
    cv2.imshow('photo', screen)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('d'):
            break

cv2.destroyAllWindows()

print("--- %s seconds ---" % (sum(t) / len(t)))
