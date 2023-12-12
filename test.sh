#!/bin/sh

python test_act.py \
  --batch_size=8 \
  --checkpoint=path/to/checkpoint.ckpt \
  --validation_root=/path/to/test_set_feat \
  --validation_labels=/path/to/test_labels.json \
