# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cv2

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def extract(frame):
    # chunk
    # dH=frame.shape[0]//4
    # dW=frame.shape[1]//4
    # akp=[]
    #
    # for ry in range(0,frame.shape[0],dH):
    #     for rx in range(0, frame.shape[1],dW):
    #         # find the keypoints with ORB
    #         img_chunk = frame[ry:ry+dH, rx:rx+dW]
    #         kp = orb.detect(img_chunk, None)
    #         # compute the descriptors with ORB
    #         kp, des = orb.compute(img_chunk, kp)
    #         # draw only keypoints location,not size and orientation
    #         print(img_chunk.shape)
    #         for p in kp:
    #             p.pt = (p.pt[0]+rx,p.pt[1]+ry)
    #             akp.append(p)
    #             # print(p)
    #
    # kp=akp

    kp = orb.detect(frame, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(frame, kp)

    # draw only keypoints location,not size and orientation
    img_kp = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
    return img_kp

orb = cv2.ORB_create(1000)

def play_video(video):
    cap = cv2.VideoCapture(video)

    while (cap.isOpened()):
        ret, frame = cap.read()
        # print(frame.shape)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = extract(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    play_video('free_road.mp4')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
