import cv2
from numpy import ndarray
import argparse
import datetime
import os.path

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
argparser = argparse.ArgumentParser()
argparser.add_argument("-v", "--video", type=str, required=True, help="benchmark video file input")
argparser.add_argument("-g", "--ground-truth", type=str, required=True, help="ground truth file")
argparser.add_argument("-t", "--tracker", type=str, required=True, #nargs='*',
                       help="benchmark tracker type input(BOOSTING CSRT GOTURN KCF MEDIANFLOW MOSSE TLD).")
argparser.add_argument("-n", "--no-status", action='store_true', help="option to hide tracking status.")
argparser.add_argument("-s", "--split", action='store_true', help="option to split ground truth and tracking output.")
argparser.add_argument("-c", "--capture", action="store_true", help="option to export result to video(avi).")
argparser.add_argument("-b", "--verbose", action="store_true", help="option to show benchmark result in window.")
args = vars(argparser.parse_args())

# Video input, output related variables.
is_video_write = args['capture']
video_input = cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "benchmark-{}-{}-{}.avi".format(args['tracker'], os.path.basename(str(args['video'])),
                                             datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               25, (video_width, video_height)) if is_video_write else None

# Ground-truth related variables.
ground_truth_file = open(args['ground_truth'], 'r')
ground_truth_list = ground_truth_file.readlines()
ground_truth_file.close()

# Option variables.
is_tracker_tracking = True
is_status_hide = args['no_status']
is_split_frame = args['split']
is_verbose_window = args['verbose']
tracker = None

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
    if tracker is None:
        # If tracker is not initialized, create tracker based on given parameter and
        # get first coordinates from ground truth file to init as a selected ROI.
        tracker = trackers_list.get(str(args['tracker']).upper())
        (x, y, w, h) = ground_truth_list[0].split(',')
        (x, y, w, h) = (int(x), int(y), int(w), int(h))
        tracker.init(tracking_frame, (x, y, w, h))
        # Draw a ground-truth-ROI on tracking frame, which is video frame if window is not separated.
        cv2.rectangle(tracking_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    else:
        # If tracker is already initialized, update tracker and get result coordinates
        # to draw rectangle on tracking frame.
        is_tracker_tracking, (x, y, w, h) = tracker.update(tracking_frame)
        if is_tracker_tracking:
            (x, y, w, h) = (int(x), int(y), int(w), int(h))
            cv2.rectangle(tracking_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw a ground-truth rectangle based on given text file. It assume that text file has 4 numbers
    # on each line separated by comma(,). These numbers are x, y, width, height of rectangle.
    (x, y, w, h) = ground_truth_list[frame].split(',')
    (x1, y1, x2, y2) = (int(x), int(y), int(x) + int(w), int(y) + int(h))
    # Draw this ground-truth rectangle to video frame contrary to tracker output rectangle and tracking frame.
    cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Build status strings to draw on frame.
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    current_frame_string = "Current frame: {}".format(int(fps))
    tracking_target_string = "Tracker: {}".format(args['tracker'])
    tracking_status_string = "Tracking Status: {}".format("TRACKING" if is_tracker_tracking else "MISSING")

    # If hide option was set, don't draw status strings on frame. Useful when size of video is too small.
    if is_status_hide:
        print(current_frame_string, tracking_target_string, tracking_status_string, sep="_")
    else:
        cv2.rectangle(tracking_frame, (5, 25), (220, 100), (0, 0, 0), -1)
        cv2.putText(tracking_frame, current_frame_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.putText(tracking_frame, tracking_target_string, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.putText(tracking_frame, tracking_status_string, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    # If verbose option was set, draw benchmark result in window.
    if is_verbose_window:
        cv2.imshow("benchmark", video_frame)
        if is_split_frame:
            cv2.imshow("tracking", tracking_frame)

    # If video output option was set, draw ground-truth rectangle to tracking frame and write to video file.
    # It's because window is may split by option and so ground-truth rectangle, tracking output rectangle are
    # drawed on separate window. x1, y1, x2, y2 variables which are coordinates of ground-truth rectangle
    # keep their values to the end of this loop. So drawing ground-truth rectangle to tracking frame
    # won't be any problem. Remember, tracking frame is video frame when there's no split option was set.
    if is_video_write and video_writer is not None:
        cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        video_writer.write(tracking_frame)

    frame = frame + 1
    key = cv2.waitKey(30)
    if key == ord("q"):
        break

video_input.release()
cv2.destroyAllWindows()
