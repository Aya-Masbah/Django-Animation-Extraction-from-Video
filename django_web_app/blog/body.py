# body.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .clientUDP import ClientUDP

import cv2
import threading
import time
from . import global_vars
import struct

class CaptureThread(threading.Thread):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = None
        self.ret = None
        self.frame = None
        self.isRunning = False
        self.counter = 0
        self.timer = 0.0

    def run(self):
        # Ouvre le fichier vidéo
        self.cap = cv2.VideoCapture(self.video_path)
        if global_vars.USE_CUSTOM_CAM_SETTINGS:
            self.cap.set(cv2.CAP_PROP_FPS, global_vars.FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, global_vars.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, global_vars.HEIGHT)

        time.sleep(1)

        print("Opened Video Capture @ %s fps" % str(self.cap.get(cv2.CAP_PROP_FPS)))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps
        self.isRunning = True

        while not global_vars.KILL_THREADS and self.isRunning:
            start_time = time.time()
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                print("End of video reached")
                self.isRunning = False
                break
            if global_vars.DEBUG:
                self.counter += 1
                if time.time() - self.timer >= 3:
                    print("Capture FPS: ", self.counter / (time.time() - self.timer))
                    self.counter = 0
                    self.timer = time.time()
            # Introduit un délai pour imiter le comportement de capture en temps réel
            elapsed_time = time.time() - start_time
            time.sleep(max(0, frame_delay - elapsed_time))
        self.cap.release()
        global_vars.KILL_THREADS = True  # Indique aux autres threads de s'arrêter


class BodyThread(threading.Thread):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.data = ""
        self.dirty = True
        self.pipe = None
        self.timeSinceCheckedConnection = 0
        self.timeSincePostStatistics = 0

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        self.setup_comms()

        capture = CaptureThread(self.video_path)
        capture.start()

        with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.5,
                          model_complexity=global_vars.MODEL_COMPLEXITY, static_image_mode=False,
                          enable_segmentation=True) as pose:

            while not global_vars.KILL_THREADS and not capture.isRunning:
                print("Waiting for video and capture thread.")
                time.sleep(0.5)
            print("Beginning capture")

            while not global_vars.KILL_THREADS and capture.isRunning:
                ti = time.time()

                # Récupère les données du thread de capture
                ret = capture.ret
                image = capture.frame

                # Transformations et traitement de l'image
                if ret:
                    image = cv2.flip(image, 1)
                    image.flags.writeable = global_vars.DEBUG

                    # Détection
                    results = pose.process(image)
                    tf = time.time()

                    # Rendu des résultats
                    if global_vars.DEBUG:
                        if time.time() - self.timeSincePostStatistics >= 1:
                            print("Theoretical Maximum FPS: %f" % (1 / (tf - ti)))
                            self.timeSincePostStatistics = time.time()

                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2,
                                                                             circle_radius=4),
                                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                             circle_radius=2),
                                                      )
                        cv2.imshow('Body Tracking', image)
                        cv2.waitKey(3)

                    # Prépare les données pour la transmission
                    self.data = ""
                    i = 0
                    if results.pose_world_landmarks:
                        hand_world_landmarks = results.pose_world_landmarks
                        for i in range(0, 33):
                            self.data += "{}|{}|{}|{}\n".format(i, hand_world_landmarks.landmark[i].x,
                                                                hand_world_landmarks.landmark[i].y,
                                                                hand_world_landmarks.landmark[i].z)

                    self.send_data(self.data)

        if self.pipe:
            self.pipe.close()
        capture.cap.release()
        cv2.destroyAllWindows()
        pass

    def setup_comms(self):
        if not global_vars.USE_LEGACY_PIPES:
            self.client = ClientUDP(global_vars.HOST, global_vars.PORT)
            self.client.start()
        else:
            print("Using Pipes for interprocess communication (not supported on OSX or Linux).")
        pass

    def send_data(self, message):
        if not global_vars.USE_LEGACY_PIPES:
            self.client.sendMessage(message)
            pass
        else:
            # Maintain pipe connection.
            if self.pipe is None and time.time() - self.timeSinceCheckedConnection >= 1:
                try:
                    self.pipe = open(r'\\.\pipe\UnityMediaPipeBody1', 'r+b', 0)
                except FileNotFoundError:
                    print("Waiting for Unity project to run...")
                    self.pipe = None
                self.timeSinceCheckedConnection = time.time()

            if self.pipe is not None:
                try:
                    s = self.data.encode('utf-8')
                    self.pipe.write(struct.pack('I', len(s)) + s)
                    self.pipe.seek(0)
                except Exception as ex:
                    print("Failed to write to pipe. Is the unity project open?")
                    self.pipe = None
        pass
