import cv2
import argparse
import datetime
import os.path
import statistics
import random

from bbox import bbox2d
from bbox import metrics
import matplotlib.pyplot as plt


# Tracker creator dictionary.
trackers_list: dict = {"BOOSTING": cv2.TrackerBoosting_create(),
                       "CSRT": cv2.TrackerCSRT_create(),
                       "GOTURN": cv2.TrackerGOTURN_create(),
                       "KCF": cv2.TrackerKCF_create(),
                       "MEDIANFLOW": cv2.TrackerMedianFlow_create(),
                       "MOSSE": cv2.TrackerMOSSE_create(),
                       "TLD": cv2.TrackerTLD_create(),
                       "MIL": cv2.TrackerMIL_create()}

# Argument to specify options.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, required=True, help="benchmark video file input")
ap.add_argument("-g", "--ground-truth", type=str, required=True, help="ground truth file")
ap.add_argument("-t", "--tracker", type=str, required=True, nargs='*',
                help="benchmark tracker type input(BOOSTING CSRT GOTURN KCF MEDIANFLOW MOSSE TLD).")
ap.add_argument("--no-status", action='store_true', help="option to hide tracking status.")
ap.add_argument("--split", action='store_true', help="option to split ground truth and tracking output.")
ap.add_argument("--output", action="store_true", help="option to export result to video(avi).")
ap.add_argument("--benchmark-output", action="store_true", help="option to export benchmark result in txt.")
args = vars(ap.parse_args())

# Video input, output related variables.
video_input = cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "benchmark-{}-{}-{}.avi".format(args['tracker'], os.path.basename(str(args['video'])),
                                             datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               25, (video_width, video_height)) if args['output'] else None

# Ground-truth related variables.
ground_truth_file = open(args['ground_truth'], 'r')
ground_truth_list = ground_truth_file.readlines()
ground_truth_file.close()

# Option variables.
is_tracker_tracking = True
is_status_hide = args['no_status']
is_split_frame = args['split']

# Tracker variables.
iou_rates = dict()
selected_tracker_names = [n for n in map(lambda i: i.upper(), args['tracker'])]
benchmark_text_writers = []
is_write_benchmark = args['benchmark_output']
trackers = []


# Initialize IoU dictionary
for tracker_name in selected_tracker_names:
    iou_rates.update({tracker_name: []})

# Initialize text writers if option is set
if is_write_benchmark:
    for tracker_name in selected_tracker_names:
        benchmark_text_name = "benchmark-{}-{}-{}.txt".format(tracker_name, os.path.basename(str(args['video'])),
                                                              datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        benchmark_text_writers.append(open(benchmark_text_name, "w"))

# Start from 0# frame of video.
frame = 0
while video_input.isOpened():
    timer = cv2.getTickCount()
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:
        break
    tracking_frame = video_frame.copy() if is_split_frame else video_frame
    # If separated window is required, copy video frame to draw on another window.\

    # Tracking output rectangle.
    bounding_box_with_color = []
    if len(trackers) == 0:
        # If tracker is not initialized, create tracker based on given parameter and
        # get first coordinates from ground truth file to init as a selected ROI.
        for tracker_name in selected_tracker_names:
            # tracker instance, tracker name, randomly generated color(between 0 ~ 1).
            trackers.append((trackers_list.get(tracker_name), tracker_name, (random.random(), random.random(), random.random())))

        (tx, ty, tw, th) = ground_truth_list[0].split(',')
        (tx, ty, tw, th) = (int(tx), int(ty), int(tw), int(th))

        # If option is set, write benchmark bounding box result.
        for index, tracker_info in enumerate(trackers):
            tracker, name, color = tracker_info
            tracker.init(tracking_frame, (tx, ty, tw, th))
            bounding_box_with_color.append(((tx, ty, tw, th), color))
            if is_write_benchmark:
                benchmark_text_writers[index].write("{},{},{},{}\n".format(tx, ty, tw, th))
    else:
        # If tracker is already initialized, update tracker and get result coordinates
        # to draw rectangle on tracking frame.
        for index, tracker_info in enumerate(trackers):
            tracker, name, color = tracker_info
            is_tracker_tracking, (tx, ty, tw, th) = tracker.update(tracking_frame)
            (tx, ty, tw, th) = (int(tx), int(ty), int(tw), int(th))

            # If tracker has failed, write zero values to benchmark result.
            if is_tracker_tracking is False:
                tx, ty, tw, th = 0, 0, 0, 0

            bounding_box_with_color.append(((tx, ty, tw, th), color))
            if is_write_benchmark:
                benchmark_text_writers[index].write("{},{},{},{}\n".format(tx, ty, tw, th))

    # Draw bounding box result.
    for (tx, ty, tw, th), color in bounding_box_with_color:
        r, g, b = [c for c in map(lambda i: i * 255, color)]
        cv2.rectangle(tracking_frame, (tx, ty), (tx+tw, ty+th), (b, g, r), 2)

    # Draw a ground-truth rectangle based on given text file. It assume that text file has 4 numbers
    # on each line separated by comma(,). These numbers are x, y, width, height of rectangle.
    (x, y, w, h) = ground_truth_list[frame].split(',')
    (x1, y1, x2, y2) = (int(x), int(y), int(x) + int(w), int(y) + int(h))
    # Draw this ground-truth rectangle to video frame contrary to tracker output rectangle and tracking frame.
    cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # ground-truth tracker is 'green' color.

    # Calculate bounding boxes IoU rate between tracker result and ground-truth.
    bboxTruth = bbox2d.BBox2D((x, y, w, h))
    for index, value in enumerate(bounding_box_with_color):
        correspond_tracker_name = selected_tracker_names[index]
        (tx, ty, tw, th), color = value
        bboxTracker = bbox2d.BBox2D((tx, ty, tw, th))
        iou_rate = int(metrics.iou_2d(bboxTracker, bboxTruth) * 100)  # % unit. floored.
        iou_rates[correspond_tracker_name].append(iou_rate)

    # Build status strings to draw on frame.
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    current_frame_string = "Current fps: {}".format(int(fps))

    # If hide option was set, don't draw status strings on frame. Useful when size of video is too small.
    if is_status_hide is False:
        cv2.rectangle(tracking_frame, (5, 25), (220, 70 + 20 * len(iou_rates)), (0, 0, 0), -1)
        cv2.putText(tracking_frame, current_frame_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

        for index, iou_rate_info in enumerate(iou_rates.items()):
            tracker_name, iou_rate_list = iou_rate_info
            tracking_iou_string = "{} IoU Rate: {}%".format(tracker_name, iou_rate_list[-1])
            cv2.putText(tracking_frame, tracking_iou_string, (10, 70 + 20 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    # If split option was set, draw benchmark result in another window.
    cv2.imshow("benchmark", video_frame)
    if is_split_frame:
        cv2.imshow("tracking", tracking_frame)

    # If video output option was set, draw ground-truth rectangle to tracking frame and write to video file.
    # It's because window is may split by option and so ground-truth rectangle, tracking output rectangle are
    # drawed on separate window. x1, y1, x2, y2 variables which are 'coordinates of ground-truth rectangle'
    # keep their values to the end of this loop. So drawing ground-truth rectangle to tracking frame
    # won't be any problem. Remember, tracking frame is video frame when there's no split option was set.
    if video_writer is not None:
        cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        video_writer.write(tracking_frame)

    frame = frame + 1
    key = cv2.waitKey(30)
    if key == ord("q"):
        break

tracker_plots = []
for tracker, name, color in trackers:
    tracker_plot = plt.plot([f for f in range(frame)], iou_rates[name], label=name, color=color)
    tracker_plots.append(tracker_plot)

plt.title("Accuracy(IoU) rate of trackers and ground-truth")
plt.ylabel('iou rate(accuracy)')
plt.xlabel('frame sequence')
plt.legend()
plt.show()

for tracker_name, iou_rate_list in iou_rates.items():
    print("{}: {}%".format(tracker_name, statistics.mean(iou_rate_list)))

if video_writer is not None:
    video_writer.release()
if is_write_benchmark is not None:
    for benchmark_text_writer in benchmark_text_writers:
        benchmark_text_writer.close()
video_input.release()
cv2.destroyAllWindows()
