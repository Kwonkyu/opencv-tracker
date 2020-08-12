import cv2


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


def get_centroid(*bounding_box) -> tuple:
    bounding_box = tuple([int(value) for value in bounding_box])
    (_x, _y, _w, _h) = bounding_box
    centroid = (_x + _w / 2, _y + _h / 2)
    return centroid
