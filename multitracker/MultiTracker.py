import cv2
import argparse
import datetime
from enum import Enum
from TrackerStatus import TrackerStatus

# Init variables.
# - Argument parsing related variables.
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", type=str, help="video file input")
parser.add_argument("-t", "--tracker", type=str, nargs='*', required=True,
                    help="tracker type input(BOOSTING CSRT GOTURN KCF MEDIANFLOW MOSSE TLD MIL).")
parser.add_argument("--output", action='store_true', help="option to write result to video file.")
args = vars(parser.parse_args())
selected_tracker_name = args['tracker']


# - Decide whether multiple object mode or multiple tracker mode.
class TrackingMode(Enum):
    MULTI_TRACKER = 1
    MULTI_OBJECT = 2


if len(selected_tracker_name) > 1:
    tracking_mode = TrackingMode.MULTI_TRACKER
else:
    tracking_mode = TrackingMode.MULTI_OBJECT

# - Video(or camera) related variables.
video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "output-{}-{}.avi".format(selected_tracker_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               25, (video_width, video_height)) if args['output'] else None


# - Tracking related variables.
def get_tracker(name):
    trackers_list = {"BOOSTING": {"name": "BOOSTING",
                                  "instance": cv2.TrackerBoosting_create(),
                                  "color": (0, 0, 255)},
                     "CSRT": {"name": "CSRT",
                              "instance": cv2.TrackerCSRT_create(),
                              "color": (66, 141, 245)},
                     "GOTURN": {"name": "GOTURN",
                                "instance": cv2.TrackerGOTURN_create(),
                                "color": (66, 245, 240)},
                     "KCF": {"name": "KCF",
                             "instance": cv2.TrackerKCF_create(),
                             "color": (111, 245, 66)},
                     "MEDIANFLOW": {"name": "MEDIANFLOW",
                                    "instance": cv2.TrackerMedianFlow_create(),
                                    "color": (245, 147, 66)},
                     "MOSSE": {"name": "MOSSE",
                               "instance": cv2.TrackerMOSSE_create(),
                               "color": (181, 43, 50)},
                     "TLD": {"name": "TLD",
                             "instance": cv2.TrackerTLD_create(),
                             "color": (234, 5, 255)},
                     "MIL": {"name": "MIL",
                             "instance": cv2.TrackerMIL_create(),
                             "color": (255, 255, 255)}}
    return trackers_list[name]


trackers = []
tracking_window_name = "MultiTracker - {}".format(selected_tracker_name)

# While capturing video...
while video_input.isOpened():
    timer = cv2.getTickCount()
    # Read video frame.
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:  # If video ends, quit program.
        print("END OF VIDEO STREAM")
        break
    # Keystroke event handling.
    key = cv2.waitKey(10) & 0xFF
    if key == ord("s"):  # 's' to select region of interest(ROI) which is tracking target.
        # Clear target container.
        trackers.clear()

        # Select ROI(s) and initialize tracker(s).
        while True:
            tracking_region = cv2.selectROI(tracking_window_name, video_frame, True)
            if tracking_region == (0, 0, 0, 0):  # selectROI() returns (0, 0, 0, 0) when cancelled.
                print("Selecting ROI cancelled.")
                break
            print("ROI selected.")

            # If multi-object, single-tracker mode('MULTI_OBJECT'), append ROI to object container and continue.
            if tracking_mode is TrackingMode.MULTI_OBJECT:
                tracker_set = get_tracker(selected_tracker_name[0])
                tracker_status = TrackerStatus(tracker_set['instance'], tracker_set['name'], tracker_set['color'])
                tracker_status.init_tracker(video_frame, tracking_region)
                trackers.append(tracker_status)
                continue

            # If single-object, multi-tracker mode('MULTI_TRACKER'), create and initialize trackers with selected ROI.
            if tracking_mode is TrackingMode.MULTI_TRACKER:
                for tracker_name in selected_tracker_name:
                    tracker_set = get_tracker(tracker_name)
                    tracker_status = TrackerStatus(tracker_set['instance'], tracker_set['name'], tracker_set['color'])
                    tracker_status.init_tracker(video_frame, tracking_region)
                    trackers.append(tracker_status)
                break
    elif key == ord("c"):  # 'c' to clear trackers.
        trackers.clear()
    elif key == ord("q"):  # 'q' to quit program.
        break

    # Tracking targets if exist.
    successful_tracker = []
    failed_tracker = []
    if len(trackers) != 0:
        for tracker_status in trackers:
            tracker_status.update_tracker(video_frame)
            (cx, cy) = tracker_status.centroid
            # If tracking success, add tracker's bounding box and color to successful tracker container.
            # TODO: filtering tracking output!
            # 1. if tracker is detected as failed, disable at once and never turn back.
            # 2. even if tracker is detected as failed, wait for it to recover
            if tracker_status.is_tracker_tracking and (not tracker_status.is_tracker_jumping) \
                    and (0 < cx < video_width) and (0 < cy < video_height):
                successful_tracker.append(tracker_status)
            # If tracking fails, add tracker's name to failed tracker container.
            else:
                failed_tracker.append(tracker_status)

    # Draw successful tracking result's bounding boxes and centroid if exist
    if len(successful_tracker) != 0:
        for tracker_status in successful_tracker:
            x, y, w, h = tracker_status.bounding_box
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), tracker_status.unique_color, 2)
            cv2.circle(video_frame, tuple(tracker_status.centroid), 2, tracker_status.unique_color, 2)

    # Draw information text.
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    current_frame_string = "Current frame: {}".format(int(fps))
    tracking_target_string = "Tracking Target: {}".format("SET" if len(trackers) != 0 else "NOT SET")
    tracking_status_string = "Tracking Status: {} / {}".format(len(successful_tracker), len(trackers))
    cv2.rectangle(video_frame, (5, 25), (270, 100), (0, 0, 0), -1)
    cv2.putText(video_frame, current_frame_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.putText(video_frame, tracking_target_string, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.putText(video_frame, tracking_status_string, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

    # Draw tracker status.
    tracker_status_loc_y = 110
    cv2.rectangle(video_frame, (5, tracker_status_loc_y), (180, 120 + 20 * len(selected_tracker_name)), (0, 0, 0), -1)
    failed_tracker_names = [n for n in map(lambda t: t.instance_name, failed_tracker)]
    for tracker_name in selected_tracker_name:
        tracker_status_string = "{}({})".format(tracker_name,
                                                "FAIL" if tracker_name in failed_tracker_names else "NORMAL")
        tracker_status_string.center(15)
        tracker_status_loc_y = tracker_status_loc_y + 20
        cv2.putText(video_frame, tracker_status_string, (10, tracker_status_loc_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_tracker(tracker_name)['color'])
    cv2.imshow(tracking_window_name, video_frame)
    if video_writer is not None:
        video_writer.write(video_frame)

if video_writer is not None:
    video_writer.release()
video_input.release()
cv2.destroyAllWindows()
