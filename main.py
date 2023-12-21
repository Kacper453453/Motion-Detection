import cv2
import numpy as np

THRESH = 20
ASSIGN_VALUE = 255
ALPHA = 0.1


def update_background(cur_frame, prev_bg, alpha):
  bg = alpha * cur_frame + (1 - alpha) * prev_bg
  bg = np.uint8(bg)
  return bg


videos = ['pexels-lazar-gugleta-13905520 (720p).mp4']

path = 'videos/' + videos[3]

cap = cv2.VideoCapture(0)

background = None

while cap.isOpened():
  ret, frame = cap.read()

  frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

  if background is None:
    background = frame_gray
    continue

  diff = cv2.absdiff(background, frame_gray)
  ret, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, cv2.THRESH_BINARY)

  background = update_background(frame_gray, background, alpha=ALPHA)


  # motion_mask = fgbg.apply(frame, LEARNING_RATE)

  # background = fgbg.getBackgroundImage()


  cont,_ = cv2.findContours(motion_mask.copy(),   
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
  
  for c in cont:
    if cv2.contourArea(c) < 10_000:
      continue

    (cur_x, cur_y,cur_w, cur_h) = cv2.boundingRect(c) 
    cv2.rectangle(frame, (cur_x, cur_y), (cur_x + cur_w, cur_y + cur_h), (0, 255, 0), 3)

  #cv2.imshow('background', background)
  #cv2.imshow('Motion Mask', motion_mask)
  cv2.imshow("From the PC or Laptop webcam, this is one example of the Colour Frame:", frame)

  if cv2.waitKey(1) == ord('q'):
    break

   
cap.release()
cv2.destroyAllWindows()