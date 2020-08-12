import cv2
import datetime
import argparse
import os, os.path
from PIL import Image


argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--images", type=str, required=True, help="image files input")
argparser.add_argument("-g", "--ground-truth", type=str, required=True, help="ground truth file")
argparser.add_argument("-n", "--no-bounding-box", action='store_false',
                       help="option to not write video file. default is write video file.")
args = vars(argparser.parse_args())

image_directory = args['images']
images = [image_directory+name for name in os.listdir(args['images'])]
video_width, video_height = Image.open(images[0]).size
output_video_filename = "output-benchmark-{}.avi".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
video_writer = cv2.VideoWriter(output_video_filename, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),\
                               25, (video_width, video_height))
is_draw_bbox = args['no_bounding_box']

ground_truth_file = open(args['ground_truth'], 'r')
ground_truth_list = ground_truth_file.readlines()

for image, ground_truth in zip(images, ground_truth_list):
    (x, y, w, h) = ground_truth.split(',')
    (x1, y1, x2, y2) = (int(x), int(y), int(x)+int(w), int(y)+int(h))

    cv2img = cv2.imread(image)
    if is_draw_bbox:
        cv2.rectangle(cv2img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("test", cv2img)

    video_writer.write(cv2img)

    key = cv2.waitKey(30)
    if key == ord("q"):
        break

ground_truth_file.close()