import cv2
from enum import Enum

class tracker_types(str, Enum):
    BOOSTING = "BOOSTING",
    MIL = "MIL",
    KCF = "KCF",
    TLD = "TLD",
    MEDIANFLOW = "MEDIANFLOW",
    GOTURN = "GOTURN",
    MOSSE = "MOSSE",
    CSRT = "CSRT"

# select tracker type
tracker_type = tracker_types.BOOSTING
print(f"Using tracker: {tracker_type}")

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create() 
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create() 
if tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create() 
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create() 
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()


# init video capture and get frame for roi
use_webcam = False
filename = "movie.mp4"
frame = None
cap = None

if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(filename)
if not cap.isOpened():
    raise IOError("Cannot open capture device or file")

if use_webcam:
    while True:
        ret, frame = cap.read()
        if not ret:
            raise IOError("Cannot read capture")
        cv2.putText(frame, "Press q to take the frame for ROI", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        cv2.imshow("Frame for ROI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow("Frame for ROI")
            break
else:
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read capture")

frame_height, frame_width = frame.shape[:2]
new_frame_size = (frame_width//2, frame_height//2)
frame = cv2.resize(frame, new_frame_size)

# Initialize video writer to save the results
# output = cv2.VideoWriter(f'{tracker_type}.avi', cv2.VideoWriter_fourcc(*'XVID'), 60.0, (frame_width//2, frame_height//2), True)

# Select the bounding box in the frame and initialze the tracker
bbox = cv2.selectROI(frame, False)
ret = tracker.init(frame, bbox)
if not ret:
    raise IOError("Cannot initialize tracker")

# Start tracking
while True:
    ret, frame = cap.read()
    if not ret:
        print('something went wrong')
        break
    frame = cv2.resize(frame, new_frame_size)
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100,80),  cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    cv2.imshow("Tracking", frame)
    # output.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
# output.release()
cv2.destroyAllWindows()