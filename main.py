import numpy as np
import cv2 as cv
from PIL import Image
from limits import limits

yellow = (0, 255, 255)

capture = cv.VideoCapture(0)

while True:
    check, frame = capture.read()

    hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_limit, upper_limit = limits(color=yellow)

    mask = cv.inRange(hsv_img, lower_limit, upper_limit)
    mask_ = Image.fromarray(mask)

    boundary_box = mask_.getbbox()

    if boundary_box is not None:
        x1, y1, x2, y2 = boundary_box

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)
        cv.putText(frame, "yellow", (x1-40, y1-40), color=(0, 255, 255), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=5)

    cv.imshow('cam', frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
