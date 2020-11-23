import cv2
from extractor import Extractor


def print_hi(name):
    print(f'Hi, {name}')


fe = Extractor()


def play_video(video):
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        ret, frame = cap.read()
        matches = fe.extract(frame)

        for pt1, pt2 in matches:
            u1, v1 = map(lambda x: int(round(x)), pt1.pt)
            u2, v2 = map(lambda x: int(round(x)), pt2.pt)
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
