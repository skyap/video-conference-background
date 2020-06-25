# sudo modprobe  v4l2loopback device=2 exclusive_caps=1 card_label="Fake"
import os
import cv2
import numpy as np
import requests
import pyfakewebcam
import pafy
from datetime import datetime
import subprocess
import signal

# p = subprocess.Popen(["node","app1.js"])

cv2.setUseOptimized(True)

def get_mask(frame, bodypix_url='http://localhost:9000'):
    _, data = cv2.imencode(".jpg", frame)
    r = requests.post(
        url=bodypix_url,
        data=data.tobytes(),
        headers={'Content-Type': 'application/octet-stream'})
    mask = np.frombuffer(r.content, dtype=np.uint8)
    mask = mask.reshape((frame.shape[0], frame.shape[1]))
    return mask

def post_process_mask(mask):
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8) , iterations=1)
    mask = cv2.blur(mask.astype(float), (20,20))
    return mask

def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    return img

def hologram_effect(img):
    # add a blue tint
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    # add a halftone effect
    bandLength, bandGap = 2, 3
    for y in range(holo.shape[0]):
        if y % (bandLength+bandGap) < bandLength:
            holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)
    # combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out

def get_frame(cap, background_scaled):
    _, frame = cap.read()
    mask = None
    while mask is None:
        try:
            mask = get_mask(frame)
        except requests.RequestException:
            print("mask request failed, retrying")
    mask = post_process_mask(mask)
#     frame = hologram_effect(frame)
    inv_mask = 1-mask
    for c in range(frame.shape[2]):
        frame[:,:,c] = frame[:,:,c]*mask + background_scaled[:,:,c]*inv_mask

    return frame


# ------------------------------- setup access to the *real* webcam ------------------------------ #

cap = cv2.VideoCapture('/dev/video0')
height, width = 480, 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 60)



# --------------------------------------------- video -------------------------------------------- #

# load the virtual background video
# bg_cap = cv2.VideoCapture("bg/beach.mp4")
# bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# print(int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

# ------------------------------------------ video loop ------------------------------------------ #

# bg_cap = cv2.VideoCapture("bg/xwings.mp4")
# bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# frame_count = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))-2
# current_count = 0

# ------------------------------------------ background ------------------------------------------ #


# background = cv2.imread("bg/a_thread.jpg")
# background_scaled = cv2.resize(background, (width, height))

# ------------------------------------------ foreground ------------------------------------------ #

background = cv2.imread("bg/it.png",cv2.IMREAD_UNCHANGED)
background_scaled = cv2.resize(background, (width, height))
mask = background_scaled[...,3]
mask[mask>0] = 1
inv_mask = 1-mask
# ------------------------------------------------------------------------------------------------ #



# setup the fake camera
fake = pyfakewebcam.FakeWebcam('/dev/video2', width, height)



# frames forever
while True:

# ------------------------------------- video background loop ------------------------------------ #
    # ret,background_scaled = bg_cap.read()
    # background_scaled = cv2.resize(background_scaled,(width,height))
    # frame = get_frame(cap, background_scaled)
    # frame = cv2.flip(frame,1)

    # if current_count == frame_count:
    #     bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     current_count = 0
    # current_count+=1
# --------------------------------------- video background --------------------------------------- #

    # ret,background_scaled = bg_cap.read()
    # background_scaled = cv2.resize(background_scaled,(width,height))
    # frame = get_frame(cap, background_scaled)
    # frame = cv2.flip(frame,1)

# --------------------------------------- normal foreground -------------------------------------- #

    ret,frame = cap.read()
    # frame = cv2.flip(frame,1)
    for c in range(3):
            frame[:,:,c] = frame[:,:,c]*inv_mask + background_scaled[:,:,c]*mask

# ---------------------------------------- dark foreground --------------------------------------- #
    # frame = get_frame(cap, np.zeros_like(background_scaled))
    # frame = cv2.flip(frame,1)
    # for c in range(3):
    #         frame[:,:,c] = frame[:,:,c]*inv_mask + background_scaled[:,:,c]*mask
# ------------------------------------------------------------------------------------------------ #


    cv2.imshow("test",frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fake.schedule_frame(frame)
    keyboard = cv2.waitKey(1)
    if keyboard == ord('q') or keyboard == 27:
        break
cap.release()
cv2.destroyAllWindows()
# p.send_signal(signal.SIGINT)