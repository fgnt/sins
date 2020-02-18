import os

import lazy_dataset
import numpy as np
import torch
from padertorch import Trainer
from padertorch.contrib.je.data.transforms import Collate
from padertorch.contrib.je.data.transforms import LabelEncoder
from padertorch.contrib.je.modules.conv import CNN2d, CNN1d
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
from sins.systems.sec.model import MultinomialClassifier
from sins.utils import timestamp
from sins.systems.sad.utils import load_sections, get_sections_in_range

ex = Exp('sec-training')
storage_dir = exp_dir / 'sec' / timestamp()
observer = FileStorageObserver.create(str(storage_dir))
ex.observers.append(observer)

db = SINS()


@ex.config
def config():
    debug = False

    # Data configuration
    sound_activity_file = str(
        exp_dir / 'sad' / '2019-10-07-06-57-23' / 'ensemble_sections.json'
    )
    max_segment_length = 3.
    min_segment_length = 1.
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
    batch_size = 64
    prefetch_buffer = 2 * batch_size
    shuffle_buffer = 8 * batch_size
    max_padding_rate = 0.2
    bucket_expiration = 2000 * batch_size

    # Trainer/Model configuration
    k = 4
    trainer = {
        'model': {
            'factory': MultinomialClassifier,
            'label_key': 'scene',
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
                'out_channels': 2 * [128*k] + [8],
                'output_layer': True,
                'kernel_size': [3, 3, 1],
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0
            },
        },
        'optimizer': {
            'factory': Adam,
            'lr': 3e-4,
            'gradient_clipping': 10.,
        },
        'storage_dir': storage_dir,
        'summary_trigger': (10 if debug else 500, 'iteration'),
        'checkpoint_trigger': (100 if debug else 2500, 'iteration'),
        'stop_trigger': (1000 if debug else 100000, 'iteration'),
    }
    Trainer.get_config(trainer)

    device = 0 if torch.cuda.is_available() else 'cpu'


@ex.capture
def get_datasets(
        sound_activity_file, min_segment_length, max_segment_length,
        audio_reader, stft, mel_transform, normalizer,
        num_workers, prefetch_buffer, shuffle_buffer,
        batch_size, max_padding_rate, bucket_expiration,
        debug
):
    sections = load_sections(sound_activity_file)
    train_sections = get_sections_in_range(sections, db.train_ranges)
    validate_sections = get_sections_in_range(sections, db.validate_ranges)

    nodes = sorted(db.room_to_nodes['living'])
    sessions = prepare_sessions(
        db.sessions, room='living', include_absence=True,
        discard_other_rooms=True, discard_ambiguities=False,
        label_map_fn=lambda label:
        'social_activity' if label in ['calling', 'visit'] else label
    )
    train_data = db.get_segments(
        nodes,
        max_segment_length=30.,
        min_segment_length=min_segment_length,
        time_ranges=train_sections,
        sessions=sessions,
        session_key="scene",
    )
    validate_data = db.get_segments(
        nodes,
        max_segment_length=30.,
        min_segment_length=min_segment_length,
        time_ranges=validate_sections,
        sessions=sessions,
        session_key="scene"
    )

    if debug:
        train_data = [ds.shuffle()[:500]for ds in train_data]
        validate_data = [ds.shuffle()[:70]for ds in validate_data]
    validate_data = [
        ds[i::len(validate_data)] for i, ds in enumerate(validate_data)
    ]

    def filter_absence(example):
        return example['scene'] != 'absence'

    train_data = [
        ds.filter(filter_absence, lazy=False) for ds in train_data
    ]
    validate_data = [
        ds.filter(filter_absence, lazy=False) for ds in validate_data
    ]

    scene_encoder = LabelEncoder(
        label_key='scene', storage_dir=storage_dir, to_array=True
    )
    scene_encoder.initialize_labels(
        dataset=train_data[0], verbose=True
    )

    audio_reader = AudioReader(**audio_reader)

    train_data = [
        ds.map(scene_encoder).map(audio_reader)
        for ds in train_data
    ]

    validate_data = [
        ds.map(scene_encoder).map(audio_reader)
        for ds in validate_data
    ]

    stft = STFT(**stft)
    mel_transform = MelTransform(**mel_transform)
    normalizers = [
        Normalizer("mel_transform", name=node, **normalizer) for node in nodes
    ]
    for normalizer, ds in zip(normalizers, train_data):
        ds = ds.shuffle()[:(10 if debug else 1440)]
        ds = ds.map(stft).map(mel_transform)
        normalizer.initialize_moments(
            ds.prefetch(num_workers, prefetch_buffer), verbose=True
        )

    def finalize(example):
        segments = []
        for i, channel in enumerate(example["mel_transform"]):
            for j, onset in enumerate(
                np.arange(0., example['audio_length'] - min_segment_length, max_segment_length)
            ):
                onset_frame = int(onset/example['audio_length']*channel.shape[-2])
                offset_frame = int((onset+max_segment_length)/example['audio_length']*channel.shape[-2])
                features = channel[onset_frame:offset_frame].T
                segments.append(
                    {
                        'example_id': f"{example['example_id']}_c{i}_n{j}",
                        'timestamp': example['timestamp'] + onset,
                        'audio_length': min(max_segment_length, example['audio_length'] - onset),
                        'features': features[None].astype(np.float32),
                        'seq_len': features.shape[-1],
                        'scene': np.array(example['scene']).astype(np.int)
                    }
                )
        return segments

    def finalize_dataset(datasets, training=False):
        dataset = lazy_dataset.concatenate([
            ds.map(stft).map(mel_transform).map(normalizer)
            for ds, normalizer in zip(datasets, normalizers)
        ]).map(finalize)
        print(len(dataset))
        return dataset.shuffle(reshuffle=training).prefetch(
            num_workers, prefetch_buffer, catch_filter_exception=True
        ).unbatch().shuffle(
            reshuffle=True, buffer_size=shuffle_buffer
        ).batch_dynamic_time_series_bucket(
            batch_size, len_key="seq_len", max_padding_rate=max_padding_rate,
            expiration=bucket_expiration, drop_incomplete=training,
            sort_key="seq_len", reverse_sort=True
        ).map(Collate())

    return finalize_dataset(train_data, training=True), finalize_dataset(validate_data)


@ex.automain
def train(_run, trainer, device):
    print_config(_run)
    os.makedirs(storage_dir, exist_ok=True)
    trainer = Trainer.from_config(trainer)
    train_iter, validate_iter = get_datasets()
    if validate_iter is not None:
        trainer.register_validation_hook(
            validate_iter, metric='mean_fscore', maximize=True
        )
    trainer.train(train_iter, device=device)
