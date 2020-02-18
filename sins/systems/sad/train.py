import os

import lazy_dataset
import numpy as np
import torch
from padertorch import Trainer
from padertorch.contrib.je.data.transforms import Collate
from padertorch.contrib.je.modules.conv import CNN2d, CNN1d
from padertorch.contrib.je.modules.global_pooling import AutoPool
from padertorch.train.hooks import ModelAttributeAnnealingHook
from padertorch.train.optimizer import Adam
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from sins.database import SINS, AudioReader
from sins.database.utils import prepare_sessions
from sins.features.mel_transform import MelTransform
from sins.features.normalize import Normalizer
from sins.features.stft import STFT
from sins.paths import exp_dir
from sins.systems.sad.model import BinomialClassifier
from sins.utils import timestamp

ex = Exp('sad-training')
storage_dir = exp_dir / 'sad' / timestamp()
observer = FileStorageObserver.create(str(storage_dir))
ex.observers.append(observer)

db = SINS()


@ex.config
def config():
    debug = False

    # Data configuration
    rooms = ['living']
    nodes = [node for room in rooms for node in db.room_to_nodes[room]]
    segment_length = 60.
    holdout_validation = True
    holdout_evaluation = True
    audio_reader = {
        "source_sample_rate": 16000,
        "target_sample_rate": 16000
    }
    stft = {
        "frame_step": 320,
        "frame_length": 960,
        "fft_length": 1024,
        "pad_mode": "mean"
    }
    mel_transform = {
        "sample_rate": audio_reader["target_sample_rate"],
        "fft_length": stft["fft_length"],
        "n_mels": 64,
        "fmin": 200
    }
    normalizer = {
        "center_axis": (1,),
        "scale_axis": (1, 2),
        "storage_dir": str(storage_dir)
    }
    num_workers = 8
    batch_size = 16
    prefetch_buffer = 2 * batch_size
    shuffle_buffer = 8 * batch_size
    max_padding_rate = 0.2
    bucket_expiration = 2000 * batch_size

    # Model configuration
    k = 1
    trainer = {
        'model': {
            'factory': BinomialClassifier,
            'label_key': 'presence',
            'cnn_2d': {
                'factory': CNN2d,
                'in_channels': 1,
                'out_channels': (np.array([16, 16, 32, 32, 64, 64])*k).tolist(),
                'pool_size': [1, 2, 1, (2, 1), 1, (2, 5)],
                'output_layer': False,
                'kernel_size': 3,
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0,
            },
            'cnn_1d': {
                'factory': CNN1d,
                'in_channels': mel_transform['n_mels']*k*8,
                'out_channels': [128*k, 1],
                'kernel_size': [3, 1],
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0
            },
            'pooling': {
                'factory': AutoPool,
                'n_classes': 1,
                'trainable': False
            },
            'recall_weight': 1.
        },
        'optimizer': {
            'factory': Adam,
            'lr': 3e-4,
            'gradient_clipping': 10.,
        },
        'storage_dir': storage_dir,
        'summary_trigger': (10 if debug else 500, 'iteration'),
        'checkpoint_trigger': (100 if debug else 5000, 'iteration'),
        'stop_trigger': (1000 if debug else 30000, 'iteration'),
    }
    Trainer.get_config(trainer)

    device = 0 if torch.cuda.is_available() else 'cpu'
    alpha_final = 2.
    alpha_slope = 1 / 10000

    max_scale = 4.
    mixup = True


@ex.capture
def get_datasets(
        rooms, nodes, segment_length, holdout_validation, holdout_evaluation,
        audio_reader, stft, mel_transform, normalizer,
        num_workers, prefetch_buffer, shuffle_buffer,
        batch_size, max_padding_rate, bucket_expiration,
        max_scale, mixup, debug
):
    train_sections = db.train_ranges
    validate_sections = db.validate_ranges
    if not holdout_validation:
        train_sections.extend(db.validate_ranges)
    if not holdout_evaluation:
        train_sections.extend(db.eval_ranges)

    train_data = []
    absence_data = []
    validate_data = []
    dataset_names = []
    for room in rooms:
        cur_nodes = sorted(set(db.room_to_nodes[room]) & set(nodes))
        if not cur_nodes:
            continue
        sessions = prepare_sessions(
            db.sessions, room=room, include_absence=True,
            discard_other_rooms=True, discard_ambiguities=False,
            label_map_fn=lambda label: (False if label == "absence" else True)
        )
        absence_sessions = list(
            filter(lambda session: not session[0], sessions)
        )
        train_data.extend(db.get_segments(
            cur_nodes,
            max_segment_length=segment_length,
            min_segment_length=segment_length,
            time_ranges=train_sections,
            sessions=sessions,
            session_key="presence"
        ))
        absence_data.extend(db.get_segments(
            cur_nodes,
            max_segment_length=segment_length,
            min_segment_length=segment_length,
            time_ranges=train_sections,
            sessions=absence_sessions,
            session_key=None
        ))
        validate_data.extend(db.get_segments(
            cur_nodes,
            max_segment_length=segment_length,
            min_segment_length=segment_length,
            time_ranges=validate_sections,
            sessions=sessions,
            session_key="presence"
        ))
        dataset_names.extend(cur_nodes)

    def filter_missing_data(example):
        if not len(example["audio_path"]) > 0:
            return False
        total_samples = np.sum((
            np.array(example["audio_stop_samples"])
            - np.array(example["audio_start_samples"])
        ))
        sr = 15980
        return (int(example["audio_length"] * sr) - total_samples) < sr

    audio_reader = AudioReader(**audio_reader)

    def scale_fn(example):
        audio = example['audio_data']
        scale = (
                (1. + (max_scale - 1.) * np.random.rand(4)[:, None])
                ** (2 * np.random.choice(2, 4)[:, None] - 1.)
        )
        audio *= scale
        return example

    train_data = [
        ds.filter(filter_missing_data, lazy=False).map(audio_reader)
        for ds in train_data
    ]
    validate_data = [
        ds.filter(filter_missing_data, lazy=False).map(audio_reader)
        for ds in validate_data
    ]
    absence_data = [
        ds.filter(filter_missing_data, lazy=False).map(audio_reader)
        for ds in absence_data
    ]
    if debug:
        train_data = [ds[:100]for ds in train_data]
        validate_data = [ds[:10]for ds in validate_data]
        absence_data = [ds[:10]for ds in absence_data]

    stft = STFT(**stft)
    mel_transform = MelTransform(**mel_transform)
    normalizers = [
        Normalizer("mel_transform", name=node, **normalizer)
        for node in dataset_names
    ]
    for normalizer, ds in zip(normalizers, train_data):
        ds = ds.shuffle()[:(10 if debug else 1440)]
        ds = ds.map(stft).map(mel_transform)
        normalizer.initialize_moments(
            ds.prefetch(num_workers, prefetch_buffer), verbose=True
        )

    def maybe_mixup(train_ds, absence_ds):
        if not mixup:
            return train_ds

        def mixup_fn(example):
            audio = example['audio_data']
            if example['presence']:
                mixup_ds = train_ds
            else:
                mixup_ds = absence_ds
            mixup_idx = np.random.choice(len(mixup_ds))
            mixup_audio = mixup_ds[mixup_idx]['audio_data']
            mixup_audio = mixup_audio[..., :audio.shape[-1]]
            mixup_audio -= mixup_audio.mean(axis=-1, keepdims=True)
            audio[..., :mixup_audio.shape[-1]] += mixup_audio
            return example
        return train_ds.map(mixup_fn)

    train_data = [
        maybe_mixup(train_ds.map(scale_fn), absence_ds.map(scale_fn))
        for train_ds, absence_ds in zip(train_data, absence_data)
    ]

    def finalize(example):
        return [
            {
                'example_id': f"{example['example_id']}_c{i}",
                'timestamp': example['timestamp'],
                'audio_length': example['audio_length'],
                'features': features.T[None].astype(np.float32),
                'seq_len': features.shape[-2],
                'presence': np.array([example['presence']]).astype(np.float32)
            }
            for i, features in enumerate(example["mel_transform"])
        ]

    def finalize_dataset(datasets, training=False):
        dataset = lazy_dataset.concatenate([
            ds.map(stft).map(mel_transform).map(normalizer)
            for ds, normalizer in zip(datasets, normalizers)
        ]).map(finalize)
        print(len(dataset))
        return dataset.shuffle(reshuffle=training).prefetch(
            num_workers, prefetch_buffer
        ).unbatch().shuffle(
            reshuffle=True, buffer_size=shuffle_buffer
        ).batch_dynamic_time_series_bucket(
            batch_size, len_key="seq_len", max_padding_rate=max_padding_rate,
            expiration=bucket_expiration, drop_incomplete=training,
            sort_key="seq_len", reverse_sort=True
        ).map(Collate())

    return finalize_dataset(train_data, training=True), finalize_dataset(validate_data)


@ex.automain
def train(_run, trainer, device, alpha_slope, alpha_final):
    print_config(_run)
    os.makedirs(storage_dir, exist_ok=True)
    trainer = Trainer.from_config(trainer)
    train_iter, validate_iter = get_datasets()
    if validate_iter is not None:
        trainer.register_validation_hook(
            validate_iter, metric='fscore', maximize=True, max_checkpoints=None
        )

    trainer.register_hook(ModelAttributeAnnealingHook(
        'pooling.alpha', (100, 'iteration'),
        slope=alpha_slope, max_value=alpha_final
    ))
    trainer.train(train_iter, device=device)
