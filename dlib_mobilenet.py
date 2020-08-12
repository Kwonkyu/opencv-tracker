import numpy as np
import argparse
import cv2
import datetime
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,	help="path to input video file")
ap.add_argument("-l", "--label", help="class label we are interested in detecting + tracking")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
ap.add_argument("-n", "--nocapture", action='store_false', help="option to not write video file. default is write video file.")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# classes supported by mobilenet.

print("loading dnn model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
# using pre-trained mobilenet ssd model.

tracker = None

video_input = cv2.VideoCapture(0) if args['video'] is None else cv2.VideoCapture(args['video'])
video_width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = "output-dlib-{}.avi".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
is_video_write = args['nocapture']
video_writer = cv2.VideoWriter(video_name, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),\
                               25, (video_width, video_height)) if is_video_write else None

while video_input.isOpened():
    is_video_playing, video_frame = video_input.read()
    if is_video_playing is False:
        break

    video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    # dlib requires RGB color space.

    if tracker is None:
        blob = cv2.dnn.blobFromImage(video_frame, 0.007843, (video_width, video_height), 127.5)
        net.setInput(blob)
        detections = net.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 1])
            confidence = detections[0, 0, i, 2]
            label = CLASSES[int(detections[0, 0, i, 1])]
            # find the index of detection with the largest probability.
            # currently it tracks first object found with the largest probability.

            if confidence > args['confidence']:# and label == args['label']:
                box = detections[0, 0, i, 3:7] * np.array([video_width, video_height, video_width, video_height])
                (x1, y1, x2, y2) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x1, y1, x2, y2)
                tracker.start_track(video_frame_rgb, rect)
                # filtering detections

                cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(video_frame, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    else:
        tracker.update(video_frame_rgb)
        tracking_position = tracker.get_position()
        (x, y, w, h) = (int(tracking_position.left()), int(tracking_position.top()),
                        int(tracking_position.right()), int(tracking_position.bottom()))

        cv2.rectangle(video_frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(video_frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("dlib", video_frame)
    if is_video_write:
        video_writer.write(video_frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

video_input.release()
cv2.destroyAllWindows()