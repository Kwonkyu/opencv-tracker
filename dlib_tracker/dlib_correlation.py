import argparse
import cv2
import datetime
import dlib


# argument related variables
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to input video file")
ap.add_argument("-n", "--output", action='store_true', help="option to write video file.")
args = vars(ap.parse_args())

# tracker related variables
trackers = []

# video input, output variables.
video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "output-dlib_manual-{}.avi".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               25, (video_width, video_height)) if args['output'] else None

# other variables
tracking_window_name = "dlib_manual"

while video_input.isOpened():
    timer = cv2.getTickCount()
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:
        break

    video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    # dlib requires RGB color space.

    # Keystroke event handling.
    key = cv2.waitKey(10) & 0xFF
    if key == ord("s"):  # 's' to select region of interest(ROI) which is 'tracked'.
        trackers.clear()
        while True:
            (x, y, w, h) = cv2.selectROI(tracking_window_name, video_frame, True, False)
            if (x, y, w, h) == (0, 0, 0, 0):
                break

            (x1, y1, x2, y2) = (x, y, (x+w), (y+h))
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x1, y1, x2, y2)  # left, top, right, bottom order. which is x, y, x+w, y+h.
            tracker.start_track(video_frame_rgb, rect)
            trackers.append(tracker)
    elif key == ord("c"):  # 'c' to clear tracking.
        trackers.clear()
    elif key == 27:  # 'ESC' to quit program.
        break

    if len(trackers) > 0:
        tracked_list = []
        for tracker in trackers:
            tracker.update(video_frame_rgb)
            tracking_position = tracker.get_position()  # return predicted position of the object.
            (x1, y1, x2, y2) = (int(tracking_position.left()), int(tracking_position.top()),
                                int(tracking_position.right()), int(tracking_position.bottom()))
            tracked_list.append((x1, y1, x2, y2))

        for tracked in tracked_list:
            (x1, y1, x2, y2) = tracked
            cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Counter frames and draw text of it.
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    current_frame_string = "Current frame: {}".format(int(fps))
    tracking_target_string = "Tracking Target: {}".format("SET" if len(trackers) != 0 else "NOT SET")
    cv2.rectangle(video_frame, (5, 25), (270, 80), (0, 0, 0), -1)
    cv2.putText(video_frame, current_frame_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.putText(video_frame, tracking_target_string, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

    cv2.imshow(tracking_window_name, video_frame)
    if video_writer is not None:
        video_writer.write(video_frame)

if video_writer is not None:
    video_writer.release()
video_input.release()
cv2.destroyAllWindows()
