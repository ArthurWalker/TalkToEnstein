import cv2
import sys
import numpy
import pandas


def main():
    #loads the face cascade into memory so it’s ready for use
    faceCascde = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    while True:
        # Cature 1 frame
        ret, frame = video_capture.read()

        # Change the frame to gray color
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Capture the face
        faces = faceCascde.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors=5, minSize=(30,30), flags = cv2.CASCADE_SCALE_IMAGE)

        # Draw a rectangle around the faces:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Display the result
        cv2.imshow('Video',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()