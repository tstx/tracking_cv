# https://broutonlab.com/blog/opencv-object-tracking
import cv2 # opencv-contrib-python
import os
import time
'''
Use CSRT when you need higher object tracking accuracy and can tolerate slower FPS throughput
Use KCF when you need faster FPS throughput but can handle slightly lower object tracking accuracy
Use MOSSE when you need pure speed
'''
tracker_dict = {"BOOSTING": cv2.legacy.TrackerBoosting_create(),
                "MIL": cv2.TrackerMIL_create(),
                "KCF": cv2.TrackerKCF_create(),
                "TLD": cv2.legacy.TrackerTLD_create(),
                "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create(),
                "MOSSE": cv2.legacy.TrackerMOSSE_create(),
                "CSRT": cv2.TrackerCSRT_create()}

def get_tracker(tracker_type):
    if tracker_type not in tracker_dict:
        raise ValueError("Tracker type not found in tracker_dict")
    return tracker_dict[tracker_type]

write_results = True
if write_results:
    out_folder = 'out/out_' + str(int(time.time())) + '_CV'
    os.makedirs(out_folder, exist_ok=True)

for tracker_type in tracker_dict:
    # if tracker_type != "MOSSE":
    #     continue
    tracker = get_tracker(tracker_type)
    print(f"Tracker type: {tracker_type}")

    # tracker_type = "MOSSE"
    # tracker = get_tracker(tracker_type)

    # init video capture and get frame for roi
    use_webcam = False
    filename = "./movie.mp4"
    filename = "/home/tim/Videos/DSC_6343.MOV"
    frame = None
    cap = None

    if use_webcam:
        cap = cv2.VideoCapture(0)
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
        cap = cv2.VideoCapture(filename)
        ret, frame = cap.read()
        if not ret:
            raise IOError("Cannot read capture")

    frame_height, frame_width = frame.shape[:2]
    frame_size = (frame_width//2, frame_height//2)
    frame_size = (frame_width, frame_height)
    if frame_size != (frame.shape[1], frame.shape[0]):
        frame = cv2.resize(frame, frame_size)

    # Initialize video writer to save the results
    if write_results:
        video_writer = cv2.VideoWriter(out_folder + f'/{tracker_type}_{int(time.time())}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60.0, frame_size, True)

    # Select the bounding box in the frame and initialze the tracker
    # bbox = cv2.selectROI(frame, False)
    # # write bbox to file
    # with open(f'bbox.txt', 'w') as f:
    #     f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
    # read bbox from file
    with open(f'bbox.txt', 'r') as f:
        bbox = [int(x) for x in f.readline().split(',')]

    ret = tracker.init(frame, bbox)

    # Start tracking
    while True:
        ret, frame = cap.read()
        if not ret:
            print('frame read failed (reached end?)')
            break
        if frame_size != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, frame_size)
        timer = cv2.getTickCount()
        ret, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80),  cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        cv2.imshow("Tracking", frame)
        if write_results:
            video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    if write_results: video_writer.release()
    cv2.destroyAllWindows()