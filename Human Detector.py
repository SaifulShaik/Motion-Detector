



#########################################
#       MADE BY SAIFUL SHAIK            #
#           HUMAN DETECTOR              #
#########################################






# ** MAKE SURE YOU HAVE IMPORTED, CREATED AND INSTALLED NESSESARY TOOLS **
# OpenCV needs to be installed and activated for cv2 to work
import cv2
# For the date and time
from datetime import datetime

cap = cv2.VideoCapture(0)

# Background Subtraction
fgbg = cv2.createBackgroundSubtractorMOG2()

# Minimum contour area thresholds
min_contour_area_fg = 3000  # for frame differencing
min_contour_area_bg = 5000  # for background subtraction

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.5  # Font size chnager

# weither if its occupided or not
occupied = False

while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Frame Differencing
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Background Subtraction
    fgmask = fgbg.apply(frame1)
    contours_bg, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < min_contour_area_fg:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for contour in contours_bg:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < min_contour_area_bg:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
        occupied = True

    if occupied:
        cv2.putText(frame1, '[+] STATUS: Occupied', (10, 50), font, font_size, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame1, '[-] STATUS: Not Occupied', (10, 50), font, font_size, (0, 255, 0), 2, cv2.LINE_AA)



    # Display date and time at the bottom
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame1, current_time, (10, frame1.shape[0] - 10), font, font_size, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Motion Detection", frame1)
    occupied = False


    # PESS 1 TO EXIT OUT OF THE TAB
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        break


cap.release()
cv2.destroyAllWindows()
