# confidence: 0.8 is recommended.
# recommended tracker: KCF.
# 0.1v: detect object at first frame and set tracker on it.
# 0.2v: use centroid to tell detected object is already tracked or not.

import numpy as np
import argparse
import cv2
import os
import datetime
import json

import utils
import TrackerStatus


# Arguments to specify options.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to input image")
ap.add_argument("-y", "--yolo-json", required=True, help="json file about YOLO setting")
ap.add_argument("-t", "--tracker", required=True, help="tracker to track detected objects.")
ap.add_argument("-o", "--output", action='store_true', help="option to write output to video file.")
args = vars(ap.parse_args())

# trackers = []
trackers = dict()
tracker_class = str(args['tracker']).upper()

# Video input, output variables.
video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "output-detect_and_track-{}-{}.avi".format(tracker_class, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
is_video_write = args['output']
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               25, (video_width, video_height)) if is_video_write else None

# import YOLO settings from json.
with open(args['yolo_json']) as yolojson:
    yolo_env = json.load(yolojson)

# YOLO variables.
coco_label_path = os.path.sep.join([yolo_env["yolo-directory"], yolo_env["coco-names"]])
yolo_weight_path = os.path.sep.join([yolo_env["yolo-directory"], yolo_env["yolo-weights"]])
yolo_config_path = os.path.sep.join([yolo_env["yolo-directory"], yolo_env["yolo-cfg"]])
coco_labels = open(coco_label_path).read().strip().split("\n")
yolo_confidence = yolo_env["yolo-confidence"]
yolo_nms_threshold = yolo_env["yolo-nms-threshold"]

# Load YOLO models which are trained by Darknet team.
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weight_path)
# If opencv with CUDA, DNN_TARGET_CUDA is available.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# A getLayerNames() function gives the name of all layers of the network.
layer_name = net.getLayerNames()
# A getUnconnectedOutLayers() function gives the names of the unconnected output layers
# which are essentially the last layers of the network.
layer_name = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def yolo(image, dnn_net, layer_names, bboxes_container, confidences_container, classIDs_container):
    # Convert image to input blob for the neural network.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (320, 320), swapRB=True, crop=False)  # 320, 416, 608?
    dnn_net.setInput(blob)
    # Get a list of predicted bounding boxes as the network's output layers.
    layer_output = dnn_net.forward(layer_names)

    for output in layer_output:
        for detection in output:
            # Get index(argmax) of highest probability class on detected object.
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # If the probability of class is higher than minimal confidence, add bounding box.
            if confidence > yolo_confidence:
                # YOLO returns the center x, y coordinates of bounding box.
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (cx, cy, w, h) = box.astype("int")
                # So we need to calculate the upper left x, y coordinate of bounding box.
                bboxes_container.append([int(cx - (w / 2)), int(cy - (h / 2)), int(w), int(h)])
                confidences_container.append(float(confidence))
                classIDs_container.append(classID)

    # YOLO doesn't do NMS on detections.
    indexes_container = cv2.dnn.NMSBoxes(bboxes, confidences, yolo_confidence, yolo_nms_threshold)
    return indexes_container


while True:
    # Read each frame of video to detect objects.
    is_video_okay, video_frame = video_input.read()

    bboxes = []
    confidences = []
    classIDs = []
    indexes = yolo(video_frame, net, layer_name, bboxes, confidences, classIDs)

    # If there're detected objects, add to tracker.
    detected_objects_window = video_frame.copy()
    if len(indexes) > 0:
        for i in indexes.flatten():
            # A pandas.Index.flatten() function returns a copy of the array into one dimension.
            (x, y, w, h) = bboxes[i]
            cv2.rectangle(detected_objects_window, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "{}: {:.4f}".format(coco_labels[int(classIDs[i])], confidences[i])
            cv2.putText(detected_objects_window, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("detected objects by YOLO", detected_objects_window)
    if utils.wait_key("q"):
        print("Detection result accepted.")
        break
    else:
        print("Detection result refused.")
        continue

for i in indexes.flatten():
    # Add trackers for the number of detected objects and init each tracker with each object.
    tracker = utils.get_tracker(tracker_class)
    tracker.init(video_frame, tuple(bboxes[i]))
    # trackers.append(tracker)
    trackers.update({tracker: coco_labels[int(classIDs[i])]})

current_frame = 0
while video_input.isOpened():
    current_frame = current_frame + 1
    current_tracking_object_size = 0
    timer = cv2.getTickCount()
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:
        print("END OF VIDEO STREAM.")
        break

    # detect object by YOLO every nth frame.
    if current_frame % 15 == 0:
        current_frame = 0
        bboxes = []
        confidences = []
        classIDs = []
        indexes = yolo(video_frame, net, layer_name, bboxes, confidences, classIDs)

        # If there're detected objects, add to tracker.
        trackers = dict()
        if len(indexes) > 0:
            for i in indexes.flatten():
                tracker = utils.get_tracker(tracker_class)
                tracker.init(video_frame, tuple(bboxes[i]))
                trackers.update({tracker: coco_labels[int(classIDs[i])]})

    # for tracker in trackers:
    for tracker, object_type in trackers.items():
        is_tracker_tracking, (x, y, w, h) = tracker.update(video_frame)
        if is_tracker_tracking:
            current_tracking_object_size = current_tracking_object_size + 1
            # trackers.remove(tracker) << need to recover, not delete!
            x1, y1, x2, y2 = int(x), int(y), int(x) + int(w), int(y) + int(h)
            cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(video_frame, object_type, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    current_fps_string = "Current fps: {}".format(int(fps))
    tracking_status_string = "Tracking: {} objects".format(current_tracking_object_size)
    tracker_status_string = "Tracker: {}".format(tracker_class)

    cv2.rectangle(video_frame, (5, 25), (220, 100), (0, 0, 0), -1)
    cv2.putText(video_frame, current_fps_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(video_frame, tracking_status_string, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(video_frame, tracker_status_string, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    cv2.imshow("detect and track", video_frame)
    if is_video_write:
        video_writer.write(video_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        utils.wait_key("s")

video_input.release()
cv2.destroyAllWindows()
