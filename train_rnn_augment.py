import cv2
import numpy as np
import pickle
from augment import get_uber_video

def split_frame(frame):
    gray_hand = frame[:64,:64]
    depth_hand = frame[:64,64:]
    gray_main = frame[64:,:64]
    depth_main = frame[64:,64:]

    hand = np.stack([gray_hand, depth_hand], axis=2)
    main = np.stack([gray_main, depth_main], axis=2)

    return hand, main

def pad_frames(vid, segment_length):
    diff = segment_length - len(vid)
    before = diff // 2
    after = diff - before
    return [np.zeros(vid[0].shape)] * before + vid + [np.zeros(vid[0].shape)] * after

def logit(num):
    out = np.zeros(20)
    out[num-1] = 1
    return out

def read_segments(base_dir, sample_num, interval_size, x_offset, y_offset, rot):
    labels_path = "{}/{}/Sample{:04d}_labels.csv".format(base_dir, "train", sample_num)
    segments = []

    x_range = range(-x_offset, x_offset+1, 1) if x_offset != 0 else [0]
    y_range = range(-y_offset, y_offset+1, 1) if y_offset != 0 else [0]
    rotations = range(-rot, rot, 1) if rot != 0 else [0]

    for x_off in x_range:
        frames = get_uber_video(base_dir, sample_num, (64, 64), x_off, 0)
        with open(labels_path, 'r') as f:
            for l in f:
                [label,start,end] = [int(n) for n in l.split(',')]
                hand, main = zip(*[split_frame(f) for f in frames[start:end+1]])
                hand, main = list(hand), list(main)
                if len(hand) < interval_size:
                    hand = pad_frames(hand, interval_size)
                if len(main) < interval_size:
                    main = pad_frames(main, interval_size)
                segments.append((hand, main, label))

    for y_off in y_range:
        frames = get_uber_video(base_dir, sample_num, (64, 64), 0, y_off)
        with open(labels_path, 'r') as f:
            for l in f:
                [label,start,end] = [int(n) for n in l.split(',')]
                hand, main = zip(*[split_frame(f) for f in frames[start:end+1]])
                hand, main = list(hand), list(main)
                if len(hand) < interval_size:
                    hand = pad_frames(hand, interval_size)
                if len(main) < interval_size:
                    main = pad_frames(main, interval_size)
                segments.append((hand, main, label))

    return segments

def get_intervals(segment, interval_size):
    hand, main, label = segment
    intervals = []

    for i in range(0, len(hand) - interval_size - 1):
        intervals.append((np.stack(hand[i:i+interval_size], axis=0),
                          np.stack(main[i:i+interval_size], axis=0),
                          logit(label)))

    return intervals

def get_batches(base_dir, size, interval_size, start, end, x_offset=0, y_offset=0, rot=0, infinite=False):
    sample_num = start
    segments = []
    segment = None
    batch = []
    intervals = []

    while True:
        if (len(batch) == size):
            yield list(zip(*batch))
            batch = []
        if sample_num > end:
            if infinite:
                sample_num = 1
            else:
                if (len(batch) > 0):
                    yield list(zip(*batch))
                return
        if len(segments) == 0:
            segments = read_segments(base_dir, sample_num, interval_size, x_offset, y_offset, rot)
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

if __name__ == '__main__':
    get_batch_size()
