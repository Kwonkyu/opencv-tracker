import cv2
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to input video file")
ap.add_argument("-n", "--nocapture", action='store_false', help="option to not write video file. default is write video file.")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_name = "output-hogsvm-{}.avi".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
is_video_write = args['nocapture']
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),\
                               25, (video_width, video_height)) if is_video_write else None


while video_input.isOpened():
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:
        break

    #video_frame = imutils.resize(video_frame, width=min(400, video_width))
    (rects, weights) = hog.detectMultiScale(video_frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(video_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(video_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("hogsvm", video_frame)
    if is_video_write:
        video_writer.write(video_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

video_input.release()
cv2.destroyAllWindows()