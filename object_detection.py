import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image")
ap.add_argument("-v", "--video", help="path to input video")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("loading dnn model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

if args['video'] is None:
    video_input = cv2.VideoCapture(0)
else:
    video_input = cv2.VideoCapture(args['video'])

while video_input.isOpened():
    #print("loading an image to blob...")
    #image = cv2.imread(args["image"])
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:  # If video ends, quit program.
        print("END OF VIDEO STREAM")
        break

    (image_height, image_width) = video_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(video_frame, (300, 300)), 0.007843, (300, 300), (104.0, 177.0, 123.0))
    #blob = cv2.dnn.blobFromImage(cv2.resize(video_frame, (600, 600)))

    #print("computing object detection...")
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([image_width, image_height, image_width, image_height])
            (x, y, w, h) = box.astype("int")

            text = "{} : {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(video_frame, (x, y), (w, h), COLORS[idx], 2)
            y = y - 10 if y - 10 > 10 else y + 10
            cv2.putText(video_frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[idx], 2)

    cv2.imshow("OUTPUT", video_frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

video_input.release()
cv2.destroyAllWindows()