import cv2

# webcam_capture_loop(".")

def colorConvert(image):
  return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


filename = "/home/tim/Videos/DSC_6342.MOV"
cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture(0)


ret, frame = cap.read()
gray_frame_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#todo: blur before absdiff?
frame_count = 0
frame_max = 2000
while (True):
  frame_count += 1
  # get image
  ret, frame = cap.read()

  # Converting frame to grayscale
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Calculating Absolute Difference between Current Frame and Median Frame
  dframe = cv2.absdiff(gray_frame, gray_frame_prev)
  # dframe = cv2.dilate(dframe, np.ones((5, 5)), 1)
  gray_frame_prev = gray_frame # todo remove and use median frame for diff (background)

  # Applying Gaussian Blur to reduce noise
  blur_frame = cv2.GaussianBlur(dframe, (11,11), 0)

  # Binarizing frame - Thresholding
  ret, threshold_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Identifying Contours
  (contours, _ ) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Drawing Boundary Boxes for each Contour
  # cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
  for i in contours:
    x, y, width, height = cv2.boundingRect(i)
    cv2.rectangle(frame, (x,y), (x + width, y + height), (123,0,255), 2)
    
  # Displaying the resulting frame
  cv2.imshow('frame (press q to quit)', frame)
  # Press 'q' to exit
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # video_writer.write(cv2.resize(frame, (640,480)))
