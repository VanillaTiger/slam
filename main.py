import cv2
from extractor import Extractor
import numpy as np

H = 480
W = 854
F=200
#1540 check if square of 2/2
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

def print_hi(name):
    print(f'Hi, {name}')


fe = Extractor(K)

def play_video(video):
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        ret, frame = cap.read()
        matches = fe.extract(frame)
        print(len(matches)," matches")


        for pt1, pt2 in matches:
            u1, v1 = fe.denormalize(pt1)
            u2, v2 = fe.denormalize(pt2)
            cv2.circle(frame, (u1, v1), color=(0, 255, 0), radius=2)
            cv2.line(frame, (u1, v1), (u2, v2), color=(255, 0, 0))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Slam')
    play_video('free_road.mp4')
