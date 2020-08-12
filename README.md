# opencv-tracker
Implementation of OpenCV's built-in trackers and usage.

## MULTITRACKER.py

- BOOSTING
- MIL
- MOSSE
- CSRT
- TLD
- MEDIANFLOW
- KCF
- GOTURN

By using MultiTracker object, we can get various trackers to track same object and compare each tracker's detection accuracy.

### usage

python MULTITRACKER.py [-h] [-v VIDEO] -t [TRACKER [TRACKER ...]] [-n]

- -h, --help: print usage and argument informations.
- -v, --video: video input to select and track object. If not given, use built-in Web cam instead.
- -t, --tracker: select trackers to apply. You can select multiple trackers listed above by passing their name separated by whitespace between them. This option must be given.
- -n, --nocapture: this script automatically generates video file and write output. If you don't want to generate output and just watch how these trackers track, use this option.

## Other Scripts...

information here.