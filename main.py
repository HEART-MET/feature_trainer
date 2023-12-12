import pytorch_lightning as pl
from argparse import ArgumentParser
from feature_trainer import FeatureTrainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    parser = ArgumentParser()

    parser.add_argument('--train_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--train_labels', default='', type=str, help='Path to training labels')
    parser.add_argument('--validation_root', default='', type=str, help='Root path of validation videos')
    parser.add_argument('--validation_labels', default='', type=str, help='Path to validation labels')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model_path', default='', type=str, help='path to trained model')
    parser.add_argument('--num_classes', default=22, type=int, help='number of classes')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='loss',
        mode='min',
    )
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    model = FeatureTrainer(args)

    trainer.fit(model)


if __name__ == '__main__':
    main()

