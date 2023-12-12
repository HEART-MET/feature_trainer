import torch.utils.data as data
import torch
import numpy as np
import os
import glob
import json

def make_dataset(root, labels_file):

    video_feat_files = sorted(glob.glob(root + '/*.npy'))

    classes = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as fp:
            labels = json.load(fp)
        for key, val in labels.items():
            classes[key] = val


    frames_per_video = []
    class_labels = []
    features = []
    samples = {}
    vid_names = []
    for vid in video_feat_files:
        feat = np.load(vid)
        num_frames = feat.shape[0]
        frames_per_video.append(num_frames)
        vid_name = os.path.basename(vid)[:-3] + 'mp4'
        if classes:
            class_labels.append(classes[vid_name])
        else:
            class_labels.append(0)
        features.append(feat)
        vid_names.append(vid_name)

    samples['num_frames_per_video'] = frames_per_video
    samples['labels'] = class_labels
    samples['video_feat'] = features
    samples['video_paths'] = vid_names

    return samples


class FeatureDataset(data.Dataset):
    def __init__(self, root, labels, num_classes=20):
        self.root = root
        samples = make_dataset(self.root, labels)
        if len(samples['video_feat']) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        self.samples = samples
        self.frames_per_video = np.cumsum(self.samples['num_frames_per_video'])
        self.num_classes = num_classes

    def __getitem__(self, index):
        # get video id of requested frame
        video_idx = np.argmax(self.frames_per_video > index)
        if video_idx == 0:
            frame_idx = index
        else:
            frame_idx = index - self.frames_per_video[video_idx - 1]

        label = self.samples['labels'][video_idx]
        label_one_hot_encoded = np.zeros(self.num_classes, dtype=np.float32)
        label_one_hot_encoded[label] = 1
        label_one_hot_encoded = torch.from_numpy(label_one_hot_encoded)
        feat = self.samples['video_feat'][video_idx][frame_idx]

        return feat, label_one_hot_encoded, video_idx

    def __len__(self):
        return self.frames_per_video[-1]

def main():
    dataset = FeatureDataset('/media/tosh/Elements/Cascade2023/activity_recognition/training_set_feat', '/media/tosh/Elements/Cascade2023/activity_recognition/training_labels.json')
    feat, label, video_id = dataset[0]
    print(feat.shape)
    print(label)
    print(video_id)


if __name__ == "__main__":
    main()
