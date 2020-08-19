import numpy as np
import argparse
import cv2
import datetime
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to input video file")
ap.add_argument("-n", "--output", action='store_true', help="option to write video file.")
args = vars(ap.parse_args())

tracker = None
tracking_window_name = "dlib_manual"
video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "output-dlib_manual-{}.avi".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               25, (video_width, video_height)) if args['output'] else None

while video_input.isOpened():
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:
        break

    video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    # dlib requires RGB color space.

    # Keystroke event handling.
    key = cv2.waitKey(10) & 0xFF
    if key == ord("s"):  # 's' to select region of interest(ROI) which is 'tracked'.
        (x, y, w, h) = cv2.selectROI(tracking_window_name, video_frame, True, False)
        (x1, y1, x2, y2) = (x, y, (x+w), (y+h))

        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(x1, y1, x2, y2)
        tracker.start_track(video_frame_rgb, rect)
    elif key == ord("c"):  # 'c' to clear tracking.
        tracker = None
    elif key == 27:  # 'ESC' to quit program.
        break

    if tracker is not None:
        tracker.update(video_frame_rgb)
        tracking_position = tracker.get_position()
        (x, y, w, h) = (int(tracking_position.left()), int(tracking_position.top()),
                        int(tracking_position.right()), int(tracking_position.bottom()))

        cv2.rectangle(video_frame, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imshow(tracking_window_name, video_frame)
    if video_writer is not None:
        video_writer.write(video_frame)

if video_writer is not None:
    video_writer.release()
video_input.release()
cv2.destroyAllWindows()