import cv2
import sys
import os.path

# http://anime.udp.jp/data/lbpcascade_animeface.xml からダウンロードして同じディレクトリに配置
cascade_file = "lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(cascade_file)

def detect(image):
    global cascade
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        yield(x, y, w, h)

input = cv2.VideoCapture(sys.argv[1])
basename, _ = os.path.splitext(os.path.basename(sys.argv[1]))

print(basename)

i = 0
count = 0
while True:
    ret, frame = input.read()
    if ret == True:
        count += 1

        if count % 15 == 0:
            for (x,y, w,h) in detect(frame):
                if w > 100:
                    crop = frame[y:y+h, x:x+w]
                    #path = u'face-%s-%04d.jpg' % (basename, i)
                    path = 'face/%s-%04d.jpg' % (basename, i)
                    print(path)
                    cv2.imwrite(path, crop)
                    i += 1
    else:
        break