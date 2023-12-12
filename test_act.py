import pytorch_lightning as pl
from argparse import ArgumentParser
import pdb
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import json
from torchvision import transforms
import cv2
import json
from feature_trainer import FeatureTrainer


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--train_labels', default='', type=str, help='Path to training labels')
    parser.add_argument('--validation_root', default='', type=str, help='Root path of validation videos')
    parser.add_argument('--validation_labels', default='', type=str, help='Path to validation labels if available')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model_path', default='', type=str, help='path to trained model')
    parser.add_argument('--num_classes', default=22, type=int, help='number of classes')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint file')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    device = 'cuda:0'

    model = FeatureTrainer.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.hparams.batch_size = args.batch_size
    model.hparams.n_threads = args.n_threads
    model.hparams.validation_root = args.validation_root
    model.hparams.validation_labels = args.validation_labels
    model.eval()

    val_loader = model.val_dataloader()
    total = 0

    output = {}
    results = {}
    all_labels = {}
    final_result = {}

    current_video = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels, vidx = batch
            inputs = inputs.to(device)
            batch = inputs, labels, vidx

            per_frame_logits, predictions = model(batch)

            vidx = vidx.numpy()
            labels = labels.numpy()
            for bidx, vv in enumerate(vidx):
                if vv in results:
                    results[vv].append(per_frame_logits[bidx].detach().cpu().numpy())
                else:
                    results[vv] = []
                    results[vv].append(per_frame_logits[bidx].detach().cpu().numpy())
                if vv not in all_labels:
                    all_labels[vv] = np.argmax(labels[bidx])
            if vv != current_video:
                avg_result = np.argmax(np.stack(results[current_video]).mean(axis=0))
                final_result[current_video] = avg_result
                print('vid: %s result: %d label: %d' % (os.path.basename(val_loader.dataset.samples['video_paths'][current_video]), avg_result, all_labels[current_video]))
                current_video = vv
                results.pop(vv, None)

    avg_result = np.argmax(np.stack(results[current_video]).mean(axis=0))
    final_result[current_video] = avg_result
    print('vid: %s result: %d label: %d' % (os.path.basename(val_loader.dataset.samples['video_paths'][current_video]), avg_result, all_labels[current_video]))
    current_video = vv
    results.pop(vv, None)

    recognized_classes = np.array(list(final_result.values()))
    ground_truth = np.array(list(all_labels.values()))
    result_dict = dict([(os.path.basename(val_loader.dataset.samples['video_paths'][idx]), int(final_result[idx])) for idx in list(final_result.keys())])
    output_dir = '.'
    with open(os.path.join(output_dir, 'submission.json'), 'w') as fp:
        json.dump(result_dict, fp)

    # if labels are available
    if os.path.exists(args.validation_labels):
        true_pos = [i for i, j in zip(recognized_classes, ground_truth) if i == j]
        tpr = len(true_pos) / len(ground_truth)
        print('true positive rate: %.3f' % tpr)

if __name__ == '__main__':
    main()
