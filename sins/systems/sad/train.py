"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train print_config
python -m padertorch.contrib.examples.wavenet.train
"""
import os
from collections import defaultdict

import lazy_dataset
import numpy as np
import tensorboardX
import torch
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from sins.database import SINS, AudioReader
from sins.database.utils import prepare_sessions
from sins.features.mel_transform import MelTransform
from sins.features.normalize import Normalizer
from sins.features.stft import STFT
from sins.paths import exp_dir
from sins.systems.modules import CNN2d, CNN1d, AutoPool
from sins.systems.sad.model import BinomialClassifier
from sins.systems.utils import Collate, batch_to_device
from sins.utils import timestamp
from torch.optim import Adam

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
    model = {
        'label_key': 'presence',
        'cnn_2d': {
            'in_channels': 1,
            'hidden_channels': (np.array([16, 16, 32, 32, 64, 64])*k).tolist(),
            'pool_size': [1, 2, 1, (2, 1), 1, (2, 5)],
            'num_layers': 6,
            'out_channels': None,
            'kernel_size': 3,
            'norm': 'batch',
            'activation': 'relu',
            'gated': False,
            'dropout': .0,
        },
        'cnn_1d': {
            'in_channels': mel_transform['n_mels']*k*8,
            'hidden_channels': 128*k,
            'out_channels': 1,
            'num_layers': 2,
            'kernel_size': [3, 1],
            'norm': 'batch',
            'activation': 'relu',
            'dropout': .0
        },
        'pool': {'n_classes': 1, 'trainable': False, 'detach_weights': False},
        'recall_weight': 1.
    }

    # Training configuration
    device = 0 if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    gradient_clipping = 10.
    summary_interval = 10 if debug else 100
    validation_interval = 100 if debug else 10000
    max_steps = 1000 if debug else 100000
    checkpoint_interval = validation_interval
    alpha_final = 2.
    alpha_anneal_start = 50 if debug else 5000
    alpha_anneal_stop = 100 if debug else 10000

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

    def finalize_dataset(datasets):
        dataset = lazy_dataset.concatenate([
            ds.map(stft).map(mel_transform).map(normalizer)
            for ds, normalizer in zip(datasets, normalizers)
        ])
        print(len(dataset))
        return dataset.map(finalize).shuffle(reshuffle=True).prefetch(
            num_workers, prefetch_buffer
        ).unbatch().shuffle(
            reshuffle=True, buffer_size=shuffle_buffer
        ).batch_dynamic_time_series_bucket(
            batch_size, len_key="seq_len", max_padding_rate=max_padding_rate,
            expiration=bucket_expiration, drop_incomplete=True,
            sort_key="seq_len", reverse_sort=True
        ).map(Collate())

    return finalize_dataset(train_data), finalize_dataset(validate_data)


@ex.automain
def train(
        _run, model, device, lr, gradient_clipping, max_steps,
        summary_interval, validation_interval, checkpoint_interval,
        alpha_final, alpha_anneal_start, alpha_anneal_stop
):
    print_config(_run)
    os.makedirs(storage_dir, exist_ok=True)
    train_iter, validate_iter = get_datasets()
    model = BinomialClassifier(
        cnn_2d=CNN2d(**model['cnn_2d']),
        cnn_1d=CNN1d(**model['cnn_1d']),
        pooling=AutoPool(**model['pool'])
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(device)
    model.train()
    optimizer = Adam(tuple(model.parameters()), lr=lr)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # Summary
    summary_writer = tensorboardX.SummaryWriter(str(storage_dir))

    def get_empty_summary():
        return dict(
            scalars=defaultdict(list),
            histograms=defaultdict(list),
            images=dict(),
        )

    def update_summary(review, summary):
        review['scalars']['loss'] = review['loss'].detach()
        for key, value in review['scalars'].items():
            if torch.is_tensor(value):
                value = value.cpu().data.numpy()
            summary['scalars'][key].extend(
                np.array(value).flatten().tolist()
            )
        for key, value in review['histograms'].items():
            if torch.is_tensor(value):
                value = value.cpu().data.numpy()
            summary['histograms'][key].extend(
                np.array(value).flatten().tolist()
            )
        summary['images'] = review['images']

    def dump_summary(summary, prefix, iteration):
        # write summary
        for key, value in summary['scalars'].items():
            summary_writer.add_scalar(
                f'{prefix}/{key}', np.mean(value), iteration
            )
        for key, values in summary['histograms'].items():
            summary_writer.add_histogram(
                f'{prefix}/{key}', np.array(values), iteration
            )
        for key, image in summary['images'].items():
            summary_writer.add_image(
                f'{prefix}/{key}', image, iteration
            )
        return defaultdict(list)

    # Training loop
    print('Start Training')
    alpha_slope = (alpha_final - 1.) / (alpha_anneal_stop - alpha_anneal_start)
    train_summary = get_empty_summary()
    i = 0
    best_validation_loss = np.inf
    while i < max_steps:
        for batch in train_iter:
            optimizer.zero_grad()
            # forward
            batch = batch_to_device(batch, device=device)
            model_out = model(batch)

            # backward
            review = model.review(batch, model_out)
            review['loss'].backward()
            review['histograms']['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                tuple(model.parameters()), gradient_clipping
            )
            optimizer.step()

            # update summary
            update_summary(review, train_summary)

            i += 1
            model.pooling.alpha = 1. + max(min(i, alpha_anneal_stop) - alpha_anneal_start, 0.) * alpha_slope
            if i % summary_interval == 0:
                train_summary = model.modify_summary(train_summary)
                dump_summary(train_summary, 'training', i)
                train_summary = get_empty_summary()
            if i % validation_interval == 0 and validate_iter is not None:
                print('Starting Validation')
                model.eval()
                validate_summary = get_empty_summary()
                with torch.no_grad():
                    for batch in validate_iter:
                        batch = batch_to_device(batch, device=device)
                        model_out = model(batch)
                        review = model.review(batch, model_out)
                        update_summary(review, validate_summary)
                validate_summary = model.modify_summary(validate_summary)
                dump_summary(validate_summary, 'validation', i)
                validation_loss = validate_summary["scalars"]["loss"]
                print('Finished Validation')
                if validation_loss < best_validation_loss:
                    print('New best validation loss:', validation_loss)
                    best_validation_loss = validation_loss
                    torch.save(
                        model.state_dict(),
                        storage_dir / 'ckpt-best.pth'
                    )
                else:
                    print('Validation loss:', validation_loss)
                print('Validation F-score:', validate_summary["scalars"]["fscore"])
                model.train()
            if i % checkpoint_interval == 0:
                torch.save(
                    model.state_dict(),
                    storage_dir / f'ckpt-{i}.pth'
                )
            if i >= max_steps:
                break
