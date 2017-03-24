import math
import numpy as np
import cv2
from collections import namedtuple
from typing import List, NamedTuple, Tuple
import subprocess as sp
import warnings
import pickle

DataPoint = namedtuple('DataPoint', ['gray', 'depth', 'user', 'skel', 'label'])
Coords = namedtuple('Coords', ['x', 'y'])
Joints = namedtuple('Joints', ['head', 'neck', 'left', 'right'])
Bounds = namedtuple('Bounds', ['min', 'max'])

def read_sample(base_dir, sample_num):
    train_dir = base_dir + "/train"

    color_path = "{}/Sample{:04d}_color.mp4".format(train_dir, sample_num)
    depth_path = "{}/Sample{:04d}_depth.mp4".format(train_dir, sample_num)
    user_path = "{}/Sample{:04d}_user.mp4".format(train_dir, sample_num)
    labels_path = "{}/Sample{:04d}_labels.csv".format(train_dir, sample_num)
    skel_path = "{}/Sample{:04d}_skeleton.csv".format(train_dir, sample_num)

    data = []
    color_vid = cv2.VideoCapture(color_path)
    depth_vid = cv2.VideoCapture(depth_path)
    user_vid = cv2.VideoCapture(user_path)
    i = 0

    with open(labels_path) as labels_file:
        with open(skel_path) as skels_file:
            for gesture_frames in labels_file:
                [label,frame_start,frame_end] = [int(n) for n in gesture_frames.split(',')]
                while i < frame_start:
                    point = next_data_point(color_vid, depth_vid, user_vid, skels_file, 0)
                    if (point is not None):
                        data.append(point)
                    i += 1
                while i <= frame_end:
                    point = next_data_point(color_vid, depth_vid, user_vid, skels_file, label)
                    if (point is not None):
                        data.append(point)
                    i += 1

    return data

def read_and_convert(video):
    ret, frame = video.read()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if ret else None

def next_data_point(color_vid, depth_vid, user_vid, skels_file, label):
    gray = read_and_convert(color_vid)
    depth = read_and_convert(depth_vid)
    user = read_and_convert(user_vid)

    if (gray is not None and depth is not None and user is not None):
        return DataPoint(gray=gray, depth=depth, user=user,
                         skel=get_joint_locs(skels_file.readline().split(',')),
                         label=label)
    else:
        return None

def get_joint_locs(joints: List[int]):
    for i in range(0, len(joints), 9):
        if (i // 9) not in [2,3,7,11]:
            continue
        if (i // 9) == 2:
            neckX = int(joints[i+7])
            neckY = int(joints[i+8])
        if (i // 9) == 3:
            headX = int(joints[i+7])
            headY = int(joints[i+8])
        # X and Y are reversed
        elif (i // 9) == 7:
            leftX = int(joints[i+7])
            leftY = int(joints[i+8])
        # X and Y are reversed
        elif (i // 9) == 11:
            rightX = int(joints[i+7])
            rightY = int(joints[i+8])

    return Joints(head=Coords(x=headX, y=headY),
                  neck=Coords(x=neckX, y=neckY),
                  left=Coords(x=leftX, y=leftY),
                  right=Coords(x=rightX, y=rightY))

def valid_coords(joint_locs: Joints):
    if (joint_locs.head.x == 0 and joint_locs.head.y == 0 and
        joint_locs.neck.x == 0 and joint_locs.neck.y == 0 and
        joint_locs.left.x == 0 and joint_locs.left.y == 0 and
        joint_locs.right.x == 0 and joint_locs.right.y == 0):
        return False

    return True

def get_hand_bounds(data: List[DataPoint], x_offset, y_offset):
    minX, minY, maxX, maxY = 640, 480, 0, 0
    for point in data:
        if (point.label != 0) and valid_coords(point.skel):
            minX = min(minX, point.skel.left.x + x_offset, point.skel.right.x + x_offset)
            minY = min(minY, point.skel.left.y + y_offset, point.skel.right.y + y_offset)
            maxX = max(maxX, point.skel.left.x + x_offset, point.skel.right.x + x_offset)
            maxY = max(maxY, point.skel.left.y + y_offset, point.skel.right.y + y_offset)

    if (minX, minY, maxX, maxY) == (640, 480, 0, 0):
        minX, minY, maxX, maxY = 0, 0, 640, 480

    return Bounds(min=Coords(x=minX, y=minY), max=Coords(x=maxX, y=maxY))

def crop_frame(frame: np.ndarray, bounds: Bounds):
    minX, maxX, _, maxY = bounds.min.x, bounds.max.x, bounds.min.y, bounds.max.y
    if maxX - minX > maxY:
        maxY = maxX - minX
    else:
        diff = (maxY - (maxX - minX))
        minX = max(0, minX - (diff // 2))
        maxX = min(640, (maxY + minX))

    return frame[:maxY, minX:maxX]

def write_video(frames: List[np.ndarray], name: str):
    writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'H264'), 20.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
    writer.release()

def write_video_ffmpeg(frames: List[np.ndarray], name: str):
    command = [ "ffmpeg",
        '-y', # (optional) overwrite output file if it exists
        '-vcodec', 'rawvideo',
        '-f', 'rawvideo',
        '-s', get_resolution(frames[0]), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '20', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'mpeg4',
        name ]

    with sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE) as pipe:
        for f in frames:
            pipe.stdin.write(cv2.cvtColor(f, cv2.COLOR_GRAY2RGB).tostring())

def get_resolution(frame: np.ndarray):
    return ("{}x{}".format(frame.shape[1], frame.shape[0]))

def crop_video(vid: List[np.ndarray], bounds: Bounds):
    return [crop_frame(f, bounds) for f in vid]

def get_hand(frame: np.ndarray, coords: Coords, shape: Tuple, x_offset, y_offset):
    minX = max(0, coords.x + x_offset - shape[1] * 3 // 4)
    maxX = min(frame.shape[1], minX + shape[1])
    minY = max(0, coords.y + y_offset - shape[0] * 3 // 4)
    maxY = min(frame.shape[0], minY + shape[0])

    if (maxX - minX) != shape[1]:
        if maxX == frame.shape[1]:
            minX = maxX - shape[1]
        else:
            maxX = minX + shape[1]
    if (maxY - minY) != shape[0]:
        if maxY == frame.shape[0]:
            minY = maxY - shape[0]
        else:
            maxY = minY + shape[0]

    return frame[minY:maxY, minX:maxX]

def get_left_hand_vid(data: List[np.ndarray], joints: List[Joints], shape: Tuple, x_offset, y_offset):
    return [get_hand(d[0], d[1].left, shape, x_offset, y_offset) for d in zip(data, joints)]

def get_right_hand_vid(data: List[np.ndarray], joints: List[Joints], shape: Tuple, x_offset, y_offset):
    return [get_hand(d[0], d[1].right, shape, x_offset, y_offset) for d in zip(data, joints)]

def get_higher_hand_vid(left: List[np.ndarray], right: List[np.ndarray], whole: List[np.ndarray],
                        joints: List[Joints], labels: List[int]):
    current_label = -1
    higher_hand = []
    main = []
    count_left = 0
    count_right = 0
    hand_left = []
    hand_right = []
    main_buffer = []

    for p in zip(left, right, whole, joints, labels):
        if p[4] != current_label:
            if count_left > count_right:
                higher_hand += [np.fliplr(f) for f in hand_left]
                main += [np.fliplr(f) for f in main_buffer]
            else:
                higher_hand += hand_right
                main += main_buffer
            current_label = -1
            count_left = 0
            count_right = 0
            hand_left = []
            hand_right = []
            main_buffer = []
            current_label = p[4]
        if p[3].left.y < p[3].right.y:
            count_left += 1
        else:
            count_right += 1
        hand_left.append(p[0])
        hand_right.append(p[1])
        main_buffer.append(p[2])

    if count_left > count_right:
        higher_hand += [np.fliplr(f) for f in hand_left]
        main += [np.fliplr(f) for f in main_buffer]
    else:
        higher_hand += hand_right
        main += main_buffer

    return higher_hand, main

def remove_background(vid, user_vid):
    modified_vid = []
    for i in range(len(vid)):
        modified_vid.append(np.ma.masked_array(vid[i], mask=user_vid[i] == 0, filled_value=124).filled(255))

    return modified_vid

def resize_video(vid, shape):
    return [cv2.resize(f, shape) for f in vid]

def get_uber_video(root, sample_num, shape: Tuple, x_offset, y_offset):
    data = read_sample(root, sample_num)
    #warnings.simplefilter("error")
    depth_vid = [d.depth for d in data]
    gray_vid = [d.gray for d in data]
    user_vid = [d.user for d in data]
    joints_list = [d.skel for d in data]
    labels_list = [d.label for d in data]
    uber_vid = []

    bounds = get_hand_bounds(data, x_offset, y_offset)
    cropped_gray_vid = crop_video(gray_vid, bounds)
    modified_depth_vid = remove_background(depth_vid, user_vid)
    cropped_depth_vid = crop_video(modified_depth_vid, bounds)
    cropped_user_vid = crop_video(user_vid, bounds)

    left_gray_vid = get_left_hand_vid(gray_vid, joints_list, (96, 96), x_offset, y_offset)
    right_gray_vid = get_right_hand_vid(gray_vid, joints_list, (96, 96), x_offset, y_offset)
    left_depth_vid = get_left_hand_vid(depth_vid, joints_list, (96, 96), x_offset, y_offset)
    right_depth_vid = get_right_hand_vid(depth_vid, joints_list, (96, 96), x_offset, y_offset)
    smaller_gray_vid = resize_video(cropped_gray_vid, shape)
    smaller_depth_vid = resize_video(cropped_depth_vid, shape)
    higher_gray_vid, smaller_gray_vid = get_higher_hand_vid(left_gray_vid, right_gray_vid, smaller_gray_vid, joints_list, labels_list)
    higher_depth_vid, smaller_depth_vid = get_higher_hand_vid(left_depth_vid, right_depth_vid, smaller_depth_vid, joints_list, labels_list)

    higher_gray_vid = resize_video(higher_gray_vid, shape)
    higher_depth_vid = resize_video(higher_depth_vid, shape)

    for i in range(len(depth_vid)):
        uber_frame = np.zeros((shape[0] * 2, shape[1] * 2), dtype='uint8')
        uber_frame[:shape[0],:shape[1]] = higher_gray_vid[i]
        uber_frame[:shape[0],shape[1]:2*shape[1]] = higher_depth_vid[i]
        uber_frame[shape[0]:2*shape[0],:shape[1]] = smaller_gray_vid[i]
        uber_frame[shape[0]:2*shape[1],shape[1]:2*shape[1]] = smaller_depth_vid[i]
        uber_vid.append(uber_frame)

    return uber_vid
