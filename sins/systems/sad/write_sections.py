import json
from pathlib import Path

import numpy as np
import torch
from sacred import Experiment as Exp
from sins import paths
from sins.database.database import SINS, AudioReader
from sins.features.mel_transform import MelTransform
from sins.features.normalize import Normalizer
from sins.features.stft import STFT
from sins.systems.modules import CNN2d, CNN1d, AutoPool
from sins.systems.sad.model import BinomialClassifier
from sins.systems.utils import Collate, batch_to_device

ex = Exp('sad-write_sections')


@ex.config
def config():
    debug = False
    exp_dir = str(paths.exp_dir / 'sad' / '2019-10-07-06-57-23')
    assert len(exp_dir) > 0, 'Set exp_dir on the command line.'
    checkpoint = 'best'
    with (Path(exp_dir) / '1' / 'config.json').open() as f:
        conf = json.load(f)
    max_segment_length = conf['segment_length']
    num_workers = 2
    prefetch_buffer = 4
    del conf

    device = 0 if torch.cuda.is_available() else 'cpu'
    decision_threshold = 0.5
    ensemble_threshold = 0.5


@ex.automain
def main(
        exp_dir, checkpoint, max_segment_length, num_workers, prefetch_buffer,
        device, decision_threshold, ensemble_threshold, debug
):
    exp_dir = Path(exp_dir)
    with (exp_dir / '1' / 'config.json').open() as f:
        conf = json.load(f)

    print(device)
    # load model
    model = BinomialClassifier(
        cnn_2d=CNN2d(**conf['model']['cnn_2d']),
        cnn_1d=CNN1d(**conf['model']['cnn_1d']),
        pooling=AutoPool(**conf['model']['pool'])
    )
    ckpt = torch.load(exp_dir / f'ckpt-{checkpoint}.pth')
    model.load_state_dict(ckpt)
    model.pooling.alpha = 2.0
    model.to(device)
    model.eval()

    # prepare data
    db = SINS()
    nodes = db.room_to_nodes['living']
    datasets = db.get_segments(
        nodes,
        min_segment_length=.2, max_segment_length=max_segment_length
    )

    audio_reader = AudioReader(**conf['audio_reader'])
    stft = STFT(**conf['stft'])
    mel_transform = MelTransform(**conf['mel_transform'])
    normalizers = [
        Normalizer("mel_transform", name=node, **conf['normalizer'])
        for node in db.room_to_nodes['living']
    ]
    [normalizer.initialize_moments(verbose=True) for normalizer in normalizers]

    def split(example):
        return [
            {
                "timestamp": example["timestamp"],
                "audio_length": example["audio_length"],
                "features": features.T[None].astype(np.float32),
                "seq_len": features.shape[-2]
            }
            for features in example["mel_transform"]
        ]

    datasets = [
        ds.map(audio_reader).map(stft).map(mel_transform).map(normalizer)
        .map(split).prefetch(
            num_workers=num_workers, buffer_size=prefetch_buffer
        )
        for ds, normalizer in zip(datasets, normalizers)
    ]

    # run segmentation
    last = np.zeros((len(nodes)*4, 1)).astype(np.bool)
    sections = [[] for _ in range(len(nodes) + 1)]
    cur_starts = [[] for _ in range(len(nodes) + 1)]
    cur_stops = [[] for _ in range(len(nodes) + 1)]
    stop_time = None
    with torch.no_grad():
        for i, batches in enumerate(zip(*datasets)):
            assert not any(cur_stops)
            # prepare batch
            batch = [example for batch in batches for example in batch]
            batch = batch_to_device(Collate()(batch), device)
            assert len(set(batch['timestamp'])) == 1
            timestamp = batch['timestamp'][0]
            if stop_time is not None:
                assert np.abs(timestamp - stop_time) < .1
            stop_time = timestamp + batch['audio_length'][-1]

            # perform sad
            sad, seq_len = model(batch)
            sad = sad.cpu().data.numpy()
            sad = sad.squeeze(1)

            # resample sad
            max_seq_len = seq_len.max()
            assert max_seq_len == sad.shape[-1]
            for j in range(len(sad)):
                sad[j] = sad[j][np.round(np.linspace(0., seq_len[j], max_seq_len, endpoint=False)).astype(np.int64)]

            # extend active sections
            hop_size = batch['audio_length'][0] / sad.shape[-1]
            sad = sad > decision_threshold
            sad = np.concatenate([last, sad], axis=-1)
            last = sad[:, -1:]

            def extend_sections(sad_, sections_, cur_starts_, cur_stops_):
                sad_ = (np.mean(sad_, axis=0) > ensemble_threshold).astype(np.int64)
                # on-/offset detection
                edges = sad_[1:] - sad_[:-1]
                start_frames = np.argwhere(edges >= 0.5).flatten()
                stop_frames = np.argwhere(edges <= -0.5).flatten()

                cur_starts_.extend((timestamp + start_frames*hop_size).tolist())
                cur_stops_.extend((timestamp + stop_frames*hop_size).tolist())
                while cur_stops_:
                    start = cur_starts_.pop(0)
                    stop = cur_stops_.pop(0)
                    sections_.append((start, stop))

            extend_sections(
                sad, sections[0], cur_starts[0], cur_stops[0]
            )
            for j in range(len(nodes)):
                extend_sections(
                    sad[j*4:(j + 1) * 4], sections[1 + j],
                    cur_starts[1 + j], cur_stops[1 + j]
                )

            if debug and i >= 19:
                break
    print(len(sections[0]))

    def write_sections(sections_, filename):
        with (Path(exp_dir) / filename).open('w') as f:
            json.dump(sections_, f, indent=4)
    write_sections(sections[0], 'ensemble_sections.json')
    for j in range(len(datasets)):
        write_sections(sections[1 + j], f'{nodes[j]}_sections.json')
