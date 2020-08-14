import cv2

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_GRAY = (128, 128, 128)


def wait_key(expected_key: str = "q"):
    while True:
        keystroke = cv2.waitKey(30)
        if keystroke & 0xFF == ord(expected_key):
            return True
        elif keystroke & 0xFF != 255:
            return False


# Tracker generator.
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


def get_centroid(bounding_box) -> tuple:
    (_x, _y, _w, _h) = bounding_box
    centroid = (_x + _w / 2, _y + _h / 2)
    return centroid


def clip_bounding_box(bounding_box, frame_width, frame_height):
    x, y, w, h = bounding_box
    # if coordinate x is out of bound(left)
    if x < 0:
        x = 0
    # if coordinate x is out of bound(right)
    elif frame_width < x + w:
        x = x - (w - (frame_width - x))
    # if coordinate y is out of bound(top)
    if y < 0:
        y = 0
    # if coordinate y is out of bound(bottom)
    elif frame_height < y + h:
        y = y - (h - (frame_height - y))

    return x, y, w, h
