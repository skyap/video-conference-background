from threading import Thread
import os
import cv2
import numpy as np
import requests
import pyfakewebcam
import pafy
from datetime import datetime
import time

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self,name,frame=None):
        self.frame = frame
        self.stopped = False
        self.thread = None
        self.name = name

    def start(self):
        self.thread = Thread(target=self.show, args=())
        self.thread.start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow(self.name, self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True
                cv2.destroyAllWindows()

    def stop(self):
        print("stopshow")
        self.stopped = True
        self.thread.join()

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = None

    def start(self):
        self.thread = Thread(target=self.get, args=())
        self.thread.start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        print("stopget")
        self.stopped = True
        self.thread.join()
        self.stream.release()


def threadBoth2():
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet("/dev/video0").start()
    time.sleep(2)
    video_shower = VideoShow("video",video_getter.frame).start()

#     bg_getter = VideoGet("bg/beach.mp4").start()
    bg_shower = VideoShow("beach",video_getter.frame).start()

    cps = CountsPerSec().start()

    while True:
#         if video_getter.stopped or video_shower.stopped or bg_getter.stopped or bg_shower.stopped:
        if video_getter.stopped or video_shower.stopped or bg_shower.stopped:

            video_shower.stop()
            video_getter.stop()
#             bg_getter.stop()
            bg_shower.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame

#         frame = bg_getter.frame
#         frame = putIterationsPerSec(frame, cps.countsPerSec())
        bg_shower.frame = frame


        cps.increment()

threadBoth2()