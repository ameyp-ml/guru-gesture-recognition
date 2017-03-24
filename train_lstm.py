import cv2
import numpy as np
import pickle

def read_video(sample_num):
    base_dir = "/media/amey/76D076A5D0766B6F/chalap"
    uber_path = "{}/{}/uber/Sample{:04d}".format(base_dir, "train", sample_num)
    video_path = uber_path + ".mp4"
    labels_path = uber_path + ".pkl"

    vid = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    return frames, labels

def split_frame(frame):
    gray_hand = frame[:64,:64]
    depth_hand = frame[:64,64:]
    gray_main = frame[64:,:64]
    depth_main = frame[64:,64:]

    hand = np.stack([gray_hand, depth_hand], axis=2)
    main = np.stack([gray_main, depth_main], axis=2)

    return hand, main

def pad_frames(vid):
    diff = 32 - len(vid)
    before = diff // 2
    after = diff - before
    return [np.zeros(vid[0].shape)] * before + vid + [np.zeros(vid[0].shape)] * after

def logit(num):
    out = np.zeros(20)
    out[num-1] = 1
    return out

def read_segments(sample_num):
    base_dir = "/media/amey/76D076A5D0766B6F/chalap"
    video_path = "{}/{}/uber/Sample{:04d}.mp4".format(base_dir, "train", sample_num)
    labels_path = "{}/{}/labels/Sample{:04d}.csv".format(base_dir, "train", sample_num)

    vid = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

    segments = []
    with open(labels_path, 'r') as f:
        for l in f:
            [label,start,end] = [int(n) for n in l.split(',')]
            hand, main = zip(*[split_frame(f) for f in frames[start:end+1]])
            hand, main = list(hand), list(main)
            if len(hand) < 32:
                hand = pad_frames(hand)
            if len(main) < 32:
                main = pad_frames(main)
            segments.append((hand, main, label))

    return segments

def get_intervals(segment, interval_size):
    hand, main, label = segment
    intervals = []

    for i in range(0, len(hand) - interval_size - 1):
        intervals.append((np.stack(hand[i:i+interval_size], axis=0),
                          np.stack(main[i:i+interval_size], axis=0),
                          np.repeat([logit(label)], interval_size, axis=0)))

    return intervals

def get_batches(size, interval_size, infinite=False):
    sample_num = 1
    segments = []
    segment = None
    batch = []
    intervals = []

    while True:
        if (len(batch) == size):
            yield list(zip(*batch))
            batch = []
        if sample_num > 470:
            if infinite:
                sample_num = 1
            else:
                if (len(batch) > 0):
                    yield list(zip(*batch))
                return
        if len(segments) == 0:
            segments = read_segments(sample_num)
            sample_num += 1
        if segment == None:
            segment = segments[0]
            segments = segments[1:]
        if len(intervals) == 0:
            intervals = get_intervals(segment, interval_size)
            segment = None

        rem = size - len(batch)
        batch = batch + intervals[:rem]
        intervals = intervals[rem:]
