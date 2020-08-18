import cv2
import argparse
import datetime


# Init variables.
# - Argument parsing related variables.
argparser = argparse.ArgumentParser()
argparser.add_argument("-v", "--video", type=str, help="video file input")
argparser.add_argument("-t", "--tracker", type=str, nargs='*', required=True,
                       help="tracker type input(BOOSTING CSRT GOTURN KCF MEDIANFLOW MOSSE TLD MIL).")
argparser.add_argument("--nocapture", action='store_false',
                       help="option to not write video file. default is write video file.")
args = vars(argparser.parse_args())
selected_tracker_name = args['tracker']

# - Video(or camera) related variables.
video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "output-{}-{}.avi".format(selected_tracker_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
is_video_write = args['nocapture']
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),\
                               25, (video_width, video_height)) if is_video_write else None


# - Tracking related variables.
def get_tracker(name):
    trackers_list: dict = {"BOOSTING": cv2.TrackerBoosting_create(),
                           "CSRT": cv2.TrackerCSRT_create(),
                           "GOTURN": cv2.TrackerGOTURN_create(),
                           "KCF": cv2.TrackerKCF_create(),
                           "MEDIANFLOW": cv2.TrackerMedianFlow_create(),
                           "MOSSE": cv2.TrackerMOSSE_create(),
                           "TLD": cv2.TrackerTLD_create(),
                           "MIL": cv2.TrackerMIL_create()}
    return trackers_list[name]


def get_multitracker():
    return cv2.MultiTracker_create()


trackers_color:dict = {"BOOSTING":(0, 0, 255),
                       "CSRT":(66, 141, 245),
                       "GOTURN":(66, 245, 240),
                       "KCF":(111, 245, 66),
                       "MEDIANFLOW":(245, 147, 66),
                       "MOSSE":(181, 43, 50),
                       "TLD":(234, 5, 255),
                       "MIL":(255, 255, 255)}

multi_tracker = None
tracking_targets = []
is_tracker_tracking = False
tracking_window_name = "Tracking - {}".format(selected_tracker_name)


# While capturing video...
while video_input.isOpened():
    timer = cv2.getTickCount()
    # Read video frame.
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False: # If video ends, quit program.
        print("END OF VIDEO STREAM")
        break

    # Keystroke event handling.
    key = cv2.waitKey(10) & 0xFF
    if key == ord("s"): # 's' to select region of interest(ROI) which is 'tracked'.
        tracking_targets.clear()
        while True: # select multiple targets
            roi = cv2.selectROI(tracking_window_name, video_frame, True, False)
            if roi == (0, 0, 0, 0): # If selectROI() cancelled, it returns zero tuples.
                break
            tracking_targets.append(roi)
            print("Region of Interest appended.")

        multi_tracker = get_multitracker()
        for tracker_name in selected_tracker_name:
            for target in tracking_targets:
                multi_tracker.add(get_tracker(tracker_name), video_frame, target)

    elif key == ord("c"): # 'c' to ...
        pass
    elif key == 27: # 'ESC' to quit program.
        break

    # Tracking target.
    if len(tracking_targets) != 0:
        is_tracker_tracking, updated_tracked = multi_tracker.update(video_frame)
        index = 0
        for tracker_name in selected_tracker_name:
            for target in tracking_targets:
                (x, y, w, h) = [floored_value for floored_value in map(lambda x: int(x), updated_tracked[index])]
                cv2.rectangle(video_frame, (x, y), (x + w, y + h), trackers_color.get(tracker_name), 2)
                index = index + 1

    # Counter frames and draw text of it.
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    current_frame_string = "Current frame: {}".format(int(fps))
    tracking_target_string = "Tracking Target: {}".format("SET" if len(tracking_targets) != 0 else "NOT SET")
    tracking_status_string = "Tracking Status: {}".format("TRACKING" if is_tracker_tracking else\
                                                              ("MISSING" if len(tracking_targets) != 0 else "READY"))
    cv2.rectangle(video_frame, (5, 25), (270, 100), (0, 0, 0), -1)
    cv2.putText(video_frame, current_frame_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.putText(video_frame, tracking_target_string, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.putText(video_frame, tracking_status_string, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

    cv2.rectangle(video_frame, (5, 110), (180, 280), (0, 0, 0), -1)
    y = 130
    for tracker_key in trackers_color.keys():
        cv2.putText(video_frame, tracker_key.center(15), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trackers_color[tracker_key])
        y = y + 20
    cv2.imshow(tracking_window_name, video_frame)
    if is_video_write:
        video_writer.write(video_frame)

video_input.release()
cv2.destroyAllWindows()
