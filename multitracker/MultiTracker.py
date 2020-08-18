import cv2
import argparse
import datetime
from enum import Enum

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
is_video_write = args['output']
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               25, (video_width, video_height)) if is_video_write else None


# - Tracking related variables.
def get_tracker(name):
    trackers_list: dict = {"BOOSTING": {"instance": cv2.TrackerBoosting_create(), "color": (0, 0, 255)},
                           "CSRT": {"instance": cv2.TrackerCSRT_create(), "color": (66, 141, 245)},
                           "GOTURN": {"instance": cv2.TrackerGOTURN_create(), "color": (66, 245, 240)},
                           "KCF": {"instance": cv2.TrackerKCF_create(), "color": (111, 245, 66)},
                           "MEDIANFLOW": {"instance": cv2.TrackerMedianFlow_create(), "color": (245, 147, 66)},
                           "MOSSE": {"instance": cv2.TrackerMOSSE_create(), "color": (181, 43, 50)},
                           "TLD": {"instance": cv2.TrackerTLD_create(), "color": (234, 5, 255)},
                           "MIL": {"instnace": cv2.TrackerMIL_create(), "color": (255, 255, 255)}}
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
                tracker = get_tracker(selected_tracker_name[0])
                tracker['instance'].init(video_frame, tracking_region)
                trackers.append(tracker)
                continue

            # If single-object, multi-tracker mode('MULTI_TRACKER'), create and initialize trackers with selected ROI.
            if tracking_mode is TrackingMode.MULTI_TRACKER:
                for tracker_name in selected_tracker_name:
                    trackers.append(get_tracker(tracker_name))
                for tracker in trackers:
                    tracker['instance'].init(video_frame, tracking_region)
                break
    elif key == ord("c"):  # 'c' to clear trackers.
        trackers.clear()
    elif key == ord("q"):  # 'q' to quit program.
        break

    # Tracking targets if exist.
    successful_tracker = []
    failed_tracker = []
    if len(trackers) != 0:
        for tracker in trackers:
            is_tracker_tracking, updated_bounding_box = tracker['instance'].update(video_frame)
            # Need to floor values because it's float.
            floored_bounding_box = [v for v in map(lambda i: int(i), updated_bounding_box)]
            if is_tracker_tracking:
                successful_tracker.append((floored_bounding_box, tracker['color']))
            else:
                # just in case
                failed_tracker.append(floored_bounding_box)

    # Draw tracking result bounding boxes if exist
    if len(successful_tracker) != 0:
        for tracker in successful_tracker:
            (x, y, w, h), color = tracker
            cv2.rectangle(video_frame, (x, y), (x+w, y+h), color, 2)

    # Counter frames and draw text of it.
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    current_frame_string = "Current frame: {}".format(int(fps))
    tracking_target_string = "Tracking Target: {}".format("SET" if len(trackers) != 0 else "NOT SET")
    tracking_status_string = "Tracking Status: {} / {}".format(len(successful_tracker), len(trackers))
    cv2.rectangle(video_frame, (5, 25), (270, 100), (0, 0, 0), -1)
    cv2.putText(video_frame, current_frame_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.putText(video_frame, tracking_target_string, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.putText(video_frame, tracking_status_string, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

    tracker_status_loc_y = 110
    cv2.rectangle(video_frame, (5, tracker_status_loc_y), (180, 130+20*len(selected_tracker_name)), (0, 0, 0), -1)
    for tracker_name in selected_tracker_name:
        tracker_status_loc_y = tracker_status_loc_y + 20
        cv2.putText(video_frame, tracker_name.center(15), (10, tracker_status_loc_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_tracker(tracker_name)['color'])
    cv2.imshow(tracking_window_name, video_frame)
    if is_video_write:
        video_writer.write(video_frame)

video_input.release()
cv2.destroyAllWindows()
