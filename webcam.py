import cv2

class Webcam:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
    def __del__(self):
        self.cap.release()

    def capture_loop(self, writepath, scale_factor = 1):
        idx = 0
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            cv2.imshow('WebcamCaptureLoop', frame)
            c = cv2.waitKey(1)
            if c == 27: # 27 is the ASCII code for the ESC key
                break
            elif c == ord('r'):
                #write frame to disk
                filepath=f'{writepath}/{idx}.png'
                print(f"recording... {filepath}")
                cv2.imwrite(f'{filepath}', frame)
                idx += 1
        cv2.destroyWindow('WebcamCaptureLoop')


    def get_latest_capture(self):
        ret, frame = self.cap.read()
        return frame