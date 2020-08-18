# confidence: 0.8 is recommended.
# recommended tracker: KCF.
# 0.1v: detect object at first frame and set tracker on it.
# 0.2v: use centroid to tell detected object is already tracked or not.

import argparse
import datetime
import json
import math
import os

import bbox.metrics
import cv2
import numpy as np

import utils
from TrackerStatus import TrackerStatus

# Arguments to specify options.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to input image")
ap.add_argument("-y", "--yolo-json", required=True, help="json file about YOLO setting")
ap.add_argument("-t", "--tracker", required=True, help="tracker to track detected objects.")
ap.add_argument("--output", action='store_true', help="option to write output to video file.")
ap.add_argument("--yolo-threshold", type=int, default=15, help="option to set frame period when yolo re-detects object")
ap.add_argument("--manual-skip", action="store_true", help="option to skip frame manually.")
ap.add_argument("--manual-yolo", action="store_true", help="option to check yolo detection result.")
args = vars(ap.parse_args())

# Tracker related variables.
tracker_class = str(args['tracker']).upper()
trackers = []
tracker_paths = dict()

# OpenCV related variables.
window_name_detected_object = "detected objects by YOLO"
window_name_tracking_object = "detect and track"
is_manual_skip = args['manual_skip']
is_manual_yolo = args['manual_yolo']

# Video input, output variables.
video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "output-detect_and_track-{}-{}.avi".format(tracker_class, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
is_video_write = args['output']
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               25, (video_width, video_height)) if is_video_write else None

# import YOLO settings from json.
with open(args['yolo_json']) as json_file:
    yolo_env = json.load(json_file)

# YOLO variables.
yolo_frame_threshold = args['yolo_threshold']
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


def yolo(image, dnn_net, layer_names, confidence_limit, threshold_limit):
    # Generate containers used in YOLO processing.
    bboxes_container = []
    confidences_container = []
    class_ids_container = []

    # Convert image to input blob for the neural network.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (320, 320), swapRB=True, crop=False)  # 320, 416, 608?
    dnn_net.setInput(blob)
    # Get a list of predicted bounding boxes as the network's output layers.
    layer_output = dnn_net.forward(layer_names)

    for output in layer_output:
        for detection in output:
            # Get index(argmax) of highest probability class on detected object.
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If the probability of class is higher than minimal confidence, add bounding box.
            if confidence > confidence_limit:
                # YOLO returns the center x, y coordinates of bounding box.
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (_cx, _cy, _w, _h) = box.astype("int")
                # So we need to calculate the upper left x, y coordinate of bounding box.
                bboxes_container.append([int(_cx - (_w / 2)), int(_cy - (_h / 2)), int(_w), int(_h)])
                confidences_container.append(float(confidence))
                class_ids_container.append(class_id)

    # YOLO doesn't do NMS on detections.
    indexes_container = cv2.dnn.NMSBoxes(bboxes_container, confidences_container, confidence_limit, threshold_limit)
    return bboxes_container, confidences_container, class_ids_container, indexes_container


while True:
    # Read each frame of video to detect objects.
    is_video_okay, video_frame = video_input.read()

    # Execute YOLO to detect objects.
    bounding_boxes, confidences, class_ids, indexes = yolo(video_frame, net, layer_name, yolo_confidence, yolo_nms_threshold)

    # If there're detected objects, draw bounding box of it to let user know where it is.
    detected_objects_window = video_frame.copy()
    if len(indexes) > 0:
        for i in indexes.flatten():
            # A pandas.Index.flatten() function returns a copy of the array into one dimension.
            (x, y, w, h) = bounding_boxes[i]
            cv2.rectangle(detected_objects_window, (x, y), (x + w, y + h), utils.COLOR_GREEN, 2)
            text = "{}: {:.4f}".format(coco_labels[int(class_ids[i])], confidences[i])
            cv2.putText(detected_objects_window, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, utils.COLOR_WHITE, 2)

    cv2.imshow(window_name_detected_object, detected_objects_window)
    if utils.wait_key("q"):
        print("Detection result accepted.")
        cv2.destroyWindow(window_name_detected_object)
        break
    else:
        print("Detection result refused.")
        continue

# If detection result is selected, store them in TrackerStatus object each.
for i in indexes.flatten():
    tracker = utils.get_tracker(tracker_class)
    tracker_status = TrackerStatus(tracker)
    tracker_status.init_tracker(video_frame, tuple(bounding_boxes[i]))
    tracker_status.set_target_category(coco_labels[int(class_ids[i])])
    trackers.append(tracker_status)
    tracker_paths.update({tracker_status: {'color': utils.get_random_color(), 'path': []}})

current_frame = 0
while video_input.isOpened():
    current_frame = current_frame + 1
    current_tracking_object_size = 0
    timer = cv2.getTickCount()
    yolo_results = dict()
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:
        print("END OF VIDEO STREAM.")
        break

    if current_frame % yolo_frame_threshold == 0:
        current_frame = 0
        # detect object by YOLO every nth frame.
        bounding_boxes, confidences, class_ids, indexes = yolo(video_frame, net, layer_name, yolo_confidence, yolo_nms_threshold)

        # try to find out this detected object is already tracked or not by comparing centroids.
        if len(indexes) > 0:
            for i in indexes.flatten():
                # store yolo results to draw on another image.
                x, y, w, h = [v for v in map(lambda j: int(j), bounding_boxes[i])]
                yolo_results.update({(x, y, w, h): {'confidence': int(confidences[i] * 100),
                                                    'category': coco_labels[int(class_ids[i])],
                                                    'signature': i}})

                is_this_object_new = True
                # get tracking object's centroid to compare with existing ones.
                compared_centroid = utils.get_centroid(bounding_boxes[i])
                # then compare with other trackers to find out if this detected-by-yolo object
                # is already tracked by other tracker.

                # try calculating intersection-over-union with bounding boxes
                bboxYOLO = bbox.BBox2D((x, y, w, h))

                for tracker_status in trackers:
                    # clipping bounding box inside of image(window).
                    clipped_bounding_box = utils.clip_bounding_box(tracker_status.get_bounding_box(), video_width, video_height)
                    bboxTracker = bbox.BBox2D(clipped_bounding_box)

                    iou_rate = bbox.metrics.iou_2d(bboxYOLO, bboxTracker) * 100
                    # if intersection-over-union rate is too high
                    if iou_rate > 50:
                        # then this object is already tracked by current tracker.
                        is_this_object_new = False
                        break

                    existing_centroid = tracker_status.get_centroid()
                    distance_between_centroids = math.dist(compared_centroid, existing_centroid)
                    # if centroid of tracker and yolo's bounding boxes are too close.
                    if distance_between_centroids < 30:
                        # Then this object is already tracked by current tracker.
                        is_this_object_new = False
                        break

                if is_this_object_new:
                    # yolo_results.append((x, y, w, h)) <-- if this expression is here, only shows new target.
                    tracker = utils.get_tracker(tracker_class)
                    tracker_status = TrackerStatus(tracker)
                    tracker_status.init_tracker(video_frame, tuple(bounding_boxes[i]))
                    tracker_status.set_target_category(coco_labels[int(class_ids[i])])
                    trackers.append(tracker_status)
                    tracker_paths.update({tracker_status: {'color': utils.get_random_color(), 'path': []}})

    # update tracker's tracking result
    for tracker_status in trackers:
        tracker_status.update_tracker(video_frame)

    failed_trackers = []
    # delete failed trackers every frame
    for tracker_status in trackers:
        # if tracking result is false
        if tracker_status.is_tracker_tracking() is False and tracker_status.is_tracker_threshold_limit():
            failed_trackers.append(tracker_status)
            continue

        # if centroid is out of view
        (cx, cy) = tracker_status.get_centroid()
        if not 0 < cx < video_width or not 0 < cy < video_height:
            failed_trackers.append(tracker_status)
            continue

        # if tracking result jumps too much distance
        if tracker_status.is_tracker_jumping():
            failed_trackers.append(tracker_status)
            continue

        # if tracking result is too small
        (x, y, w, h) = tracker_status.get_bounding_box()
        # if w < 30 or h < 30:
        if w * h < 1500:
            failed_trackers.append(tracker_status)
            continue

        # if IoU rate with other tracker is over 50%
        bboxCurrentTracker = bbox.BBox2D((x, y, w, h))
        for other_tracker in trackers:
            if id(other_tracker) == id(tracker_status):
                continue
            bboxOtherTracker = bbox.BBox2D((other_tracker.get_bounding_box()))
            iou_rate = bbox.metrics.iou_2d(bboxCurrentTracker, bboxOtherTracker) * 100
            if iou_rate > 50:
                failed_trackers.append(tracker_status)
                break

    # removed failed tracker and it's path.
    for failed_tracker in failed_trackers:
        tracker_paths.pop(failed_tracker)
        trackers.remove(failed_tracker)

    # draw at once so tracker's bounding boxes won't interfere with other tracker's tracking.
    for tracker_status in trackers:
        if tracker_status.is_tracker_tracking():
            current_tracking_object_size = current_tracking_object_size + 1
            # draw bounding box.
            (x, y, w, h) = tracker_status.get_bounding_box()
            cv2.rectangle(video_frame, (x, y), (x+w, y+h), utils.COLOR_GREEN, 2)
            # draw centroid point.
            (cx, cy) = tracker_status.get_centroid()
            cv2.circle(video_frame, (cx, cy), 2, utils.COLOR_GREEN, 2)
            # append and draw tracking paths
            path_color = tracker_paths[tracker_status]['color']
            tracker_path = tracker_paths[tracker_status]['path']
            # if path length hits threshold, remove most outdated path and insert latest one.
            if len(tracker_path) > 50:
                tracker_path.pop(0)
                # TODO: something to do is... not removing whole path when tracker is removed, but gradually.
            tracker_path.append(tracker_status.get_path())
            for path in tracker_path:
                cv2.line(video_frame, tuple(path[0]), tuple(path[1]), path_color, 2, cv2.LINE_AA)
            # draw object category, index number text.
            detection_status_string = "{} #{}".format(tracker_status.get_target_category(), tracker_status.get_index())
            cv2.putText(video_frame, detection_status_string, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, utils.COLOR_GREEN, 2)

    # if yolo_results exist and option is set, draw how yolo detected objects.
    if len(yolo_results) > 0 and is_manual_yolo:
        for x, y, w, h in yolo_results.keys():
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), utils.COLOR_BLUE, 2)
            cv2.circle(video_frame, (int(x+w/2), int(y+h/2)), 2, utils.COLOR_BLUE, 2)
            yolo_result_text = "{} at {}%".format(yolo_results[(x, y, w, h)]['category'],
                                                  yolo_results[(x, y, w, h)]['confidence'])
            cv2.putText(video_frame, "DETECTED BY YOLO", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, utils.COLOR_CYAN, 2)
            cv2.putText(video_frame, yolo_result_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, utils.COLOR_CYAN, 2)

    # tracking status information text.
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    current_fps_string = "Current fps: {}".format(int(fps))
    tracking_status_string = "Tracking: {} objects".format(current_tracking_object_size)
    tracker_status_string = "Tracker: {}".format(tracker_class)
    yolo_threshold_string = "{}".format(current_frame * "#").ljust(yolo_frame_threshold, "-")

    # draw tracking status information.
    cv2.rectangle(video_frame, (5, 25), (170 + 5 * yolo_frame_threshold, 120), utils.COLOR_BLACK, -1)
    cv2.putText(video_frame, current_fps_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, utils.COLOR_WHITE)
    cv2.putText(video_frame, tracking_status_string, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, utils.COLOR_WHITE)
    cv2.putText(video_frame, tracker_status_string, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, utils.COLOR_WHITE)
    cv2.putText(video_frame, yolo_threshold_string, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, utils.COLOR_WHITE)

    cv2.imshow(window_name_tracking_object, video_frame)
    if is_video_write:
        video_writer.write(video_frame)

    # if yolo_results exist and option is set, wait for user input.
    if len(yolo_results) > 0 and is_manual_yolo:
        utils.wait_key("q")

    # if option is set, wait for user input.
    if is_manual_skip:
        if utils.wait_key("q"):
            break
    else:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC input.
            break
        if key == ord("s"):  # 's' to pause.
            utils.wait_key("s")

video_input.release()
cv2.destroyAllWindows()
