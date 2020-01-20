import argparse
import time
import os
import sys
import configparser
import csv
import datetime

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import cv2


def define_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Configuration file")
    return vars(ap.parse_args())

def read_config(filename):
    print("[INFO] Reading config: {}".format(filename))
    if not os.path.isfile(filename):
        print("[ERROR] Config file \"{}\" not found.".format(filename))
        exit()
    cfg = configparser.ConfigParser()
    cfg.read(filename)
    return cfg


def save_count(filename, n):
    f = open(filename, "a")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    line = "{} , {}\n".format(timestamp, n)
    f.write(line)
    f.close()

def execute_network(image, network, layernames):

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    start2 = time.time()
    network.setInput(blob)
    outputs = network.forward(layernames)
    end2 = time.time()
    print("[INFO] YOLO  took      : %2.1f sec" % (end2-start2))
    return outputs

def load_network(network_folder):
    """
    Load the Yolo network from disk.
    https://pjreddie.com/media/files/yolov3.weights
    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

    :param network_folder: folder where network files are stored
    """
    # Derive file paths and check existance
    labelspath = os.path.sep.join([network_folder, "obj.names"])
    if not os.path.isfile(labelspath):
        print("[ERROR] Network: Labels file \"{}\" not found.".format(labelspath))
        exit()

    weightspath = os.path.sep.join([network_folder, "yolov3.weights"])
    if not os.path.isfile(weightspath):
        print("[ERROR] Network: Weights file \"{}\" not found.".format(weightspath))
        exit()

    configpath = os.path.sep.join([network_folder, "yolov3.cfg"])
    if not os.path.isfile(configpath):
        print("[ERROR] Network: Configuration file \"{}\" not found.".format(configpath))
        exit()

    # load YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    # Network storend in Darknet format
    print("[INFO] loading YOLO from disk...")
    labels = open(labelspath).read().strip().split("\n")
    network = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    names = network.getLayerNames()
    names = [names[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    return network, names, labels

def get_detected_items(layeroutputs, confidence_level, threshold, img_width, img_height):

    # initialize our lists of detected bounding boxes, confidences, and class IDs
    detected_boxes = []
    detection_confidences = []
    detected_classes = []

    for output in layeroutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > confidence_level:
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (center_x, center_y, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top left corner of the bounding box
                top_x = int(center_x - (width / 2))
                top_y = int(center_y - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                detected_boxes.append([top_x, top_y, int(width), int(height)])
                detection_confidences.append(float(confidence))
                detected_classes.append(classid)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(detected_boxes, detection_confidences, confidence_level, threshold)

    return indexes, detected_classes, detected_boxes, detection_confidences


def get_videowriter(outputfile, width, height, frames_per_sec=30):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(outputfile, fourcc, frames_per_sec, (width, height), True)
    return video_writer, frames_per_sec


def save_frame(video_writer, new_frame, count=1):
    for _ in range(0, count):
        video_writer.write(new_frame)


def get_webcamesource(webcam_id, width=640, height=480):
    print("[INFO] initialising video source...")
    video_device = cv2.VideoCapture(webcam_id)
    video_device.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video_device.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    (success, videoframe) = video_device.read()
    if not success:
        print("[ERROR] Could not read from webcam id {}".format(webcam_id))
    (height, width) = videoframe.shape[:2]
    print("[INFO] Frame W x H: {} x {}".format(width, height))
    return video_device, width, height


def get_filesource(filename):
    print("[INFO] initialising video source : {}".format(filename))
    video_device = cv2.VideoCapture(filename)
    width = int(video_device.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_device.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] Frame W x H: {} x {}".format(width, height))
    return video_device, width, height


def update_frame(image, people_indxs, class_ids, detected_boxes, conf_levels, colors, labels,
                 show_boxes, blur, box_all_objects):
    count_bell_jellyfish = 0
    count_blubber_jellyfish = 0
    count_moon_jellyfish = 0
    count_seanettle_jellyfish = 0

    if len(people_indxs) >= 1:
        # loop over the indexes we are keeping
        for i in people_indxs.flatten():
            # extract the bounding box coordinates
            (x, y, w, h) = (detected_boxes[i][0], detected_boxes[i][1], detected_boxes[i][2], detected_boxes[i][3])

            if classIDs[i] == 0:
                count_bell_jellyfish += 1
                # Blur, if required, people in the image
                if blur:
                    image = blur_area(image, max(x, 0), max(y, 0), w, h)
            if classIDs[i] == 1:
                count_blubber_jellyfish += 1
                # Blur, if required, people in the image
                if blur:
                    image = blur_area(image, max(x, 0), max(y, 0), w, h)
            if classIDs[i] == 2:
                count_moon_jellyfish += 1
                # Blur, if required, people in the image
                if blur:
                    image = blur_area(image, max(x, 0), max(y, 0), w, h)
            if classIDs[i] == 3:
                count_seanettle_jellyfish += 1
                # Blur, if required, people in the image
                if blur:
                    image = blur_area(image, max(x, 0), max(y, 0), w, h)

            # draw a bounding box rectangle and label on the frame
            if (show_boxes and classIDs[i] == 0) or box_all_objects:
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.2f}".format(labels[classIDs[i]], conf_levels[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    total_jellyfish = count_bell_jellyfish+count_blubber_jellyfish+count_moon_jellyfish+count_seanettle_jellyfish

    # write number of people in bottom corner
    text = "Bell Jellyfish: {}".format(count_bell_jellyfish)
    cv2.putText(image, text, (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    text2 = "Blubber Jellyfish: {}".format(count_blubber_jellyfish)
    cv2.putText(image, text2, (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    text3 = "Moon Jellyfish: {}".format(count_moon_jellyfish)
    cv2.putText(image, text3, (10, image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    text4 = "Sea Nettle Jellyfish: {}".format(count_seanettle_jellyfish)
    cv2.putText(image, text4, (10, image.shape[0] - 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # text5 = "Total Jellyfish: {}".format(count_blubber_jellyfish)
    # cv2.putText(image, text5, (10, image.shape[0] - 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image, total_jellyfish


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    args = define_args()
    config = read_config(args["config"])

    # Load the trained network
    (net, ln, LABELS) = load_network(config['NETWORK']['Path'])

    # Initialise video source
    webcam = (config['READER']['Webcam'] == "yes")
    if webcam:
        cam_id = int(config['READER']['WebcamID'])
        cam_width = int(config['READER']['Width'])
        cam_height = int(config['READER']['Height'])
        (cam, W, H) = get_webcamesource(cam_id, cam_width, cam_height)
    else:
        (cam, cam_width, cam_height) = get_filesource(config['READER']['Filename'])

    # determine if we need to show the enclosing boxes, etc
    network_path = config['NETWORK']['Path']
    webcam = (config['READER']['Webcam'] == "yes")
    showpeopleboxes = (config['OUTPUT']['ShowPeopleBoxes'] == "yes")
    showallboxes = (config['OUTPUT']['ShowAllBoxes'] == "yes")
    blurpeople = (config['OUTPUT']['BlurPeople'] == "yes")
    realspeed = (config['OUTPUT']['RealSpeed'] == "yes")
    nw_confidence = float(config['NETWORK']['Confidence'])
    nw_threshold = float(config['NETWORK']['Threshold'])
    countfile = config['OUTPUT']['Countfile']
    save_video = (config['OUTPUT']['SaveVideo'] == "yes")
    show_graphs = (config['OUTPUT']['ShowGraphs'] == "yes")
    print_ascii = (config['OUTPUT']['PrintAscii'] == "yes")
    buffer_size = int(config['READER']['Buffersize'])
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Initialise video ouptut writer
    if save_video:
        (writer, fps) = get_videowriter(config['OUTPUT']['Filename'], cam_width, cam_height,
                                        int(config['OUTPUT']['FPS']))
    else:
        (writer, fps) = (None, 0)

    # Create output windows, but limit on 1440x810
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', min(cam_width, 800), min(cam_height, 480))
    #cv2.resizeWindow('Video', min(cam_width, 640), min(cam_height, 360))
    cv2.moveWindow('Video', 0, 0)
    # Create plot
    if show_graphs:
        plt.ion()
        plt.figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
        df = read_existing_data(countfile)
    else:
        df = None

    # loop while true
    while True:
        start = time.time()
        # read the next frame from the webcam
        # make sure that buffer is empty by reading specified amount of frames
        for _ in (0, buffer_size):
            (grabbed, frame) = cam.read()  # type: (bool, np.ndarray)
        if not grabbed:
            break
        # Feed frame to network
        layerOutputs = execute_network(frame, net, ln)
        # Obtain detected objects, including cof levels and bounding boxes
        (idxs, classIDs, boxes, confidences) = get_detected_items(layerOutputs, nw_confidence, nw_threshold,
                                                                  cam_width, cam_height)

        # Update frame with recognised objects
        frame, npeople = update_frame(frame, idxs, classIDs, boxes, confidences, COLORS, LABELS, showpeopleboxes,
                                      blurpeople, showallboxes)
        save_count(countfile, npeople)

        # Show frame with bounding boxes on screen
        cv2.imshow('Video', frame)
        # write the output frame to disk, repeat (time taken * 30 fps) in order to get a video at real speed
        if save_video:
            frame_cnt = int((time.time()-start)*fps) if webcam and realspeed else 1
            save_frame(writer, frame, frame_cnt)

        end = time.time()
        print("[INFO] Total handling  : %2.1f sec" % (end - start))
        print("[INFO] Jellyfish in frame : {}".format(npeople))
        if print_ascii:
            print_ascii_large(str(npeople)+ (" persons" if npeople > 1 else " person"))
        # Check for exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the file pointers
    print("[INFO] cleaning up...")


    if save_video:
        writer.release()
    cam.release()
