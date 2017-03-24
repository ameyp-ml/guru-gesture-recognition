import cv2
import numpy as np
import pickle

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

def read_segments(base_dir, sample_num, interval_size):
    video_path = "{}/{}/Sample{:04d}.mp4".format(base_dir, "train", sample_num)
    labels_path = "{}/{}/Sample{:04d}.csv".format(base_dir, "train", sample_num)

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
            if len(hand) < interval_size:
                hand = pad_frames(hand, interval_size)
            if len(main) < interval_size:
                main = pad_frames(main, interval_size)
            segments.append((hand, main, label))

    return segments

def get_pretrain_intervals(segment, interval_size):
    hand, main, label = segment
    intervals = []

    for i in range(0, len(hand) - interval_size - 1):
        intervals.append((np.stack(hand[i:i+interval_size-1], axis=0),
                          np.stack(main[i:i+interval_size-1], axis=0),
                          np.hstack((np.vstack((cv2.resize(hand[i+interval_size-1][:,:,0], (16,16)),
                                               cv2.resize(hand[i+interval_size-1][:,:,1], (16, 16)))),
                                    np.vstack((cv2.resize(main[i+interval_size-1][:,:,0], (16, 16)),
                                               cv2.resize(main[i+interval_size-1][:,:,1], (16, 16))))))))

    return intervals

def get_intervals(segment, interval_size):
    hand, main, label = segment
    intervals = []

    for i in range(0, len(hand) - interval_size - 1):
        intervals.append((np.stack(hand[i:i+interval_size], axis=0),
                          np.stack(main[i:i+interval_size], axis=0),
                          logit(label)))

    return intervals

def get_pretrain_batches(base_dir, size, interval_size, start, end, infinite=False):
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
                sample_num = start
            else:
                if (len(batch) > 0):
                    yield list(zip(*batch))
                return
        if len(segments) == 0:
            segments = read_segments(base_dir, sample_num, interval_size)
            sample_num += 1
        if segment == None:
            segment = segments[0]
            segments = segments[1:]
        if len(intervals) == 0:
            intervals = get_pretrain_intervals(segment, interval_size)
            segment = None

        rem = size - len(batch)
        batch = batch + intervals[:rem]
        intervals = intervals[rem:]

def get_batches(base_dir, size, interval_size, start, end, infinite=False):
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
                sample_num = start
            else:
                if (len(batch) > 0):
                    yield list(zip(*batch))
                return
        if len(segments) == 0:
            segments = read_segments(base_dir, sample_num, interval_size)
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
