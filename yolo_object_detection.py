# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import datetime

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to input image")
ap.add_argument("-y", "--yolo-dir", required=True,	help="base path to YOLO directory")
ap.add_argument("-m", "--yolo-coco-names", required=True,	help="base path to YOLO directory")
ap.add_argument("-w", "--yolo-weights", required=True,	help="base path to YOLO directory")
ap.add_argument("-f", "--yolo-cfg", required=True,	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
ap.add_argument("-n", "--nocapture", action='store_false', help="option to not write video file. default is write video file.")
args = vars(ap.parse_args())

video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "output-YOLO-{}.avi".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
is_video_write = args['nocapture']
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),\
                               25, (video_width, video_height)) if is_video_write else None

coco_label_path = os.path.sep.join([args["yolo_dir"], args["yolo_coco_names"]])
yolo_weight_path = os.path.sep.join([args["yolo_dir"], args["yolo_weights"]])
yolo_config_path = os.path.sep.join([args["yolo_dir"], args["yolo_cfg"]])
coco_labels = open(coco_label_path).read().strip().split("\n")

print("loading yolo model...")
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weight_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # If opencv with CUDA, DNN_TARGET_CUDA is available.
# yolov3 models are trained by Darknet team.

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(coco_labels), 3), dtype="uint8")


while video_input.isOpened():
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:
        break

    layer_name = net.getLayerNames()
    layer_name = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # getUnconnectedOutLayers() gives the names of the unconnected output layers,
    # which are essentially the last layers of the network.

    blob = cv2.dnn.blobFromImage(video_frame, 1/255.0, (320, 320), swapRB=True, crop=False) # 320, 416, 608?
    # convert image to input blob for the neural network.
    net.setInput(blob)
    layer_output = net.forward(layer_name)
    # get a list of predicted bounding boxes as the network's output layers.

    bboxes = []
    confidences = []
    classIDs = []

    for output in layer_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (x, y, w, h) = box.astype("int")

                # yolo returns the center x, y coordinates of bounding box.
                x = int(x - (w / 2))
                y - int(y - (h / 2))

                bboxes.append([int(x), int(y), int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, args["confidence"], args["threshold"])
    # yolo doesn't NMS boxes.
    if len(indexes) > 0:
        for i in indexes.flatten():
            (x, y, w, h) = bboxes[i]
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(video_frame, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(coco_labels[classIDs[i]], confidences[i])
            cv2.putText(video_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("yolo", video_frame)
    if is_video_write:
        video_writer.write(video_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

video_input.release()
cv2.destroyAllWindows()