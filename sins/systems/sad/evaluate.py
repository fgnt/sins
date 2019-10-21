import json
from math import ceil
from pathlib import Path

import lazy_dataset
import numpy as np
import torch
from einops import rearrange
from sacred import Experiment as Exp
from sins import paths
from sins.database.database import SINS, AudioReader
from sins.database.utils import prepare_sessions
from sins.features.mel_transform import MelTransform
from sins.features.normalize import Normalizer
from sins.features.stft import STFT
from sins.systems.modules import CNN2d, CNN1d, AutoPool
from sins.systems.sad.model import BinomialClassifier
from sins.systems.utils import Collate, batch_to_device

ex = Exp('sad-eval')

db = SINS()


@ex.config
def config():
    exp_dir = str(paths.exp_dir / 'sad' / '2019-10-07-06-57-23')
    assert len(exp_dir) > 0, 'Set exp_dir on the command line.'
    sound_events_dir = ''
    assert len(sound_events_dir) > 0, 'Set sound_event_dir=/path/to/dcase2016_task2_train_sounds on the command line.'
    checkpoint = 'best'
    with (Path(exp_dir) / '1' / 'config.json').open() as f:
        conf = json.load(f)
    segment_length = 20.
    num_workers = 2
    prefetch_buffer = 4
    del conf

    device = 0 if torch.cuda.is_available() else 'cpu'
    mixtures_per_sound = 3
    snr = (-3., 10.)
    seed = 0

    meanfilt_sizes = [None]
    ensemble_thresholds = [None, 0.5]

    nodes = [node for node in db.room_to_nodes['living']]


def prepare_dataset(
        exp_dir, sound_events_dir, mixtures_per_sound, segment_length, snr,
        nodes, num_workers, prefetch_buffer, seed, dataset_name, prefetch=True
):
    assert Path(exp_dir).exists()
    assert Path(sound_events_dir).exists()
    npr = np.random.RandomState(seed)
    assert dataset_name in ['validate', 'eval'], dataset_name
    with (Path(exp_dir) / '1' / 'config.json').open() as f:
        conf = json.load(f)

    # prepare sounds
    sound_reader = AudioReader(
        source_sample_rate=44100, target_sample_rate=15980
    )
    sound_files = sorted(Path(sound_events_dir).glob('*.wav'))
    npr.shuffle(sound_files)
    validate_eval_split = sound_files[:len(sound_files)//2], sound_files[len(sound_files)//2:]
    sound_files = validate_eval_split[int(dataset_name == 'eval')]
    sounds = [sound_reader.read_file(str(file)) for file in sound_files]

    audio_reader = AudioReader(**conf['audio_reader'])
    stft = STFT(**conf['stft'])
    mel_transform = MelTransform(**conf['mel_transform'])
    normalizers = [
        Normalizer("mel_transform", name=node, **conf['normalizer'])
        for node in nodes
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

    sessions = prepare_sessions(
        db.sessions, room='living', include_absence=True,
        discard_other_rooms=True, discard_ambiguities=False,
        label_map_fn=lambda label: (False if label == "absence" else True)
    )
    sessions = list(filter(lambda x: not x[0], sessions))

    num_examples = mixtures_per_sound*len(sounds)
    npr.shuffle(sounds)

    sound_lengths = np.array([
        sound.shape[-1] for _ in range(mixtures_per_sound)
        for i, sound in enumerate(sounds)
    ])

    segments = db.get_segments(
        nodes,
        min_segment_length=segment_length,
        max_segment_length=segment_length,
        time_ranges=getattr(db, f'{dataset_name}_ranges'),
        sessions=sessions, session_key=None
    )
    print([len(ds) for ds in segments])
    shuffle_idx = np.arange(len(segments[0]))
    npr.shuffle(shuffle_idx)
    segments = [
        ds[shuffle_idx][:num_examples] for ds in segments
    ]

    # prepare mixtures
    onsets = npr.rand(num_examples)
    onset_samples = np.array([
        8000 + onset * (segment_length * 15980 - sound_lengths[i] - 16000)
        for i, onset in enumerate(onsets)
    ]).astype(np.int)
    offset_samples = onset_samples + sound_lengths
    if stft.pad_mode is not None:
        onset_samples += int((stft.frame_length - stft.frame_step) / 2)
        offset_samples += int((stft.frame_length - stft.frame_step) / 2)
    onset_frames = (onset_samples - stft.frame_length) / stft.frame_step
    offset_frames = np.ceil(offset_samples / stft.frame_step)
    if isinstance(snr, (list, tuple)):
        assert len(snr) == 2
        snrs = npr.rand(num_examples)*(snr[1]-snr[0]) + snr[0]
    else:
        snrs = snr * np.ones(num_examples)

    print([len(ds) for ds in segments])
    segments = [
        lazy_dataset.new({
            f'{example["example_id"]}': (i, example)
            for i, example in enumerate(ds)
        }) for ds in segments
    ]

    def prepare_mixture(args):
        i, example = args
        example = audio_reader(example)
        audio = example['audio_data']
        sound = sounds[i % len(sounds)]
        onset_sample = onset_samples[i]
        snr = snrs[i]

        scale = np.sqrt(
            audio.var(axis=-1, keepdims=True)
            / sound.var(axis=-1, keepdims=True) * 10**(snr/10)
        )
        audio[..., onset_sample:onset_sample+sound.shape[-1]] += sound * scale
        return example

    mixtures = [
        ds.map(prepare_mixture).map(stft).map(mel_transform).map(normalizer)
        .map(split)
        for ds, normalizer in zip(segments, normalizers)
    ]
    if prefetch:
        mixtures = [
            ds.prefetch(num_workers=num_workers, buffer_size=prefetch_buffer)
            for ds in mixtures
        ]

    return mixtures, onset_frames, offset_frames


def simple_sad(features, win_len=10):
    scores = features.mean(axis=(1, 2))
    tail = scores.shape[-1] % win_len
    if tail > 0:
        pad_width = scores.ndim * [(0, 0)]
        pad_width[-1] = (0, win_len - tail)
        scores = np.pad(scores, pad_width=pad_width, mode='mean')
    scores = scores.reshape((scores.shape[0], -1, win_len)).mean(-1)
    return scores


def run_scoring(datasets, model, device):
    all_nn_scores = []
    all_simple_scores = []

    with torch.no_grad():
        for i, batches in enumerate(zip(*datasets)):
            # prepare batch
            batch = Collate()([
                example for batch in batches for example in batch
            ])
            assert len(set(batch['timestamp'])) == 1

            # compute scores
            nn_scores, seq_len = model(batch_to_device(batch, device))
            max_seq_len = seq_len.max()
            assert all(seq_len == max_seq_len)
            nn_scores = nn_scores.cpu().data.numpy()
            nn_scores = nn_scores.squeeze(1)
            features = batch['features'].cpu().data.numpy()
            simple_scores = simple_sad(
                features,
                win_len=ceil(features.shape[-1] / nn_scores.shape[-1])
            )
            assert simple_scores.shape[-1] == nn_scores.shape[-1], (
                simple_scores.shape[-1], nn_scores.shape[-1]
            )
            all_nn_scores.append(nn_scores)
            all_simple_scores.append(simple_scores)
    return np.array(all_nn_scores), np.array(all_simple_scores)


def make_decision(
        scores, decision_threshold=0.5, meanfilt_size=None,
        ensemble_threshold=None
):
    if decision_threshold is not None:
        scores = scores > decision_threshold
    else:
        assert ensemble_threshold is not None, (decision_threshold, ensemble_threshold)
    if meanfilt_size is not None:
        meanfilt = np.ones(meanfilt_size) / meanfilt_size
        scores = np.apply_along_axis(
            lambda m: np.correlate(m, meanfilt, mode='same'),
            axis=-1, arr=scores
        ) > 0.5
    if ensemble_threshold is not None:
        scores = scores.mean(1, keepdims=True) >= ensemble_threshold
    return scores


def prepare_targets(scores, onset_frames, offset_frames):
    n_examples, n_frames = scores.shape[0], scores.shape[-1]
    targets = np.zeros((n_examples, 1, n_frames))
    onsets = onset_frames.astype(np.int)
    offsets = np.ceil(offset_frames).astype(np.int)
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        targets[i, :, onset:offset] = 1
    return targets


def fscore(targets, decisions):
    targets, decisions = np.broadcast_arrays(targets, decisions)
    if decisions.ndim == 3:
        decisions = rearrange(decisions, 'n c t -> (n c) t')
        targets = rearrange(targets, 'n c t -> (n c) t')

    # compute true positives
    tp = (targets * decisions).sum(-1) > 0.5  # * targets.sum(-1)
    tot = tp.size
    tp = tp.sum()

    # compute false positives
    targets = np.apply_along_axis(
        lambda m: np.correlate(m, np.ones(5), mode='same'),
        axis=-1, arr=targets
    ) > 0.
    decisions = decisions * (1 - targets)
    zeros = np.zeros(decisions.shape[:-1])[..., None]
    edges = (
        np.concatenate([decisions, zeros], axis=-1)
        - np.concatenate([zeros, decisions], axis=-1)
    )
    assert np.abs(edges.sum()) < 1e-3, np.abs(edges.sum())
    starts = edges[..., :-1] > 0.5
    stops = edges[..., 1:] < -0.5
    fp = (((1-targets) * starts)[starts] + ((1-targets) * stops)[stops]) > 0.5
    fp = fp.sum()

    p = tp / (tp+fp+1e-6)
    r = tp / tot
    f = 2*p*r / (p+r+1e-6)
    return f, p, r


def tune_decision_threshold(
        targets, scores, meanfilt_size=None, ensemble_threshold=None,
        max_candidates=100
):
    idx = np.argsort(scores.flatten())
    flat_scores = scores.flatten()[idx]
    flat_targets = np.broadcast_to(targets, scores.shape).flatten()[idx]
    candidate_idx = np.argwhere(flat_targets[1:] - flat_targets[:-1] > 0.).flatten()
    candidate_thresholds = (flat_scores[candidate_idx] + flat_scores[candidate_idx + 1]) / 2
    if len(candidate_idx) > max_candidates:
        candidate_thresholds = candidate_thresholds[
            np.linspace(0, len(candidate_idx) - 1, max_candidates).astype(np.int)
        ]
    if ensemble_threshold is not None:
        candidate_thresholds = [None] + candidate_thresholds.tolist()
    fscores = []
    for threshold in candidate_thresholds:
        decisions = make_decision(
            scores, decision_threshold=threshold,
            meanfilt_size=meanfilt_size, ensemble_threshold=ensemble_threshold
        )
        fscores.append(fscore(targets, decisions))
    f, p, r = zip(*fscores)
    return candidate_thresholds[np.argmax(f)]


@ex.automain
def main(
        exp_dir, checkpoint, sound_events_dir,
        mixtures_per_sound, segment_length, snr, nodes,
        meanfilt_sizes, ensemble_thresholds,
        num_workers, prefetch_buffer, device, seed
):
    assert all([node in db.room_to_nodes['living'] for node in nodes])
    exp_dir = Path(exp_dir)
    with (exp_dir / '1' / 'config.json').open() as f:
        conf = json.load(f)

    print('Device:', device)
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

    pool_sizes = [
        pool_size[1] if isinstance(pool_size, (list, tuple)) else pool_size
        for pool_size in (model._cnn_2d.pool_sizes + model._cnn_1d.pool_sizes)
    ]
    total_pool_size = np.prod(pool_sizes)

    (validate_mixtures, onset_frames, offset_frames) = prepare_dataset(
        exp_dir, sound_events_dir, mixtures_per_sound, segment_length, snr,
        nodes, num_workers, prefetch_buffer, seed, dataset_name='validate'
    )
    nn_validate_scores, simple_validate_scores = run_scoring(
        validate_mixtures, model, device
    )
    validate_targets = prepare_targets(
        nn_validate_scores,
        onset_frames / total_pool_size, offset_frames / total_pool_size
    )
    print(
        nn_validate_scores.shape,
        simple_validate_scores.shape,
        validate_targets.shape
    )
    (eval_mixtures, onset_frames, offset_frames) = prepare_dataset(
        exp_dir, sound_events_dir, mixtures_per_sound, segment_length, snr,
        nodes, num_workers, prefetch_buffer, seed, dataset_name='eval'
    )
    nn_eval_scores, simple_eval_scores = run_scoring(
        eval_mixtures, model, device
    )
    eval_targets = prepare_targets(
        nn_eval_scores,
        onset_frames / total_pool_size, offset_frames / total_pool_size
    )
    print(
        nn_eval_scores.shape,
        simple_eval_scores.shape,
        eval_targets.shape
    )

    def tune_and_eval(
            scores, targets, decision_thresholds=(),
            medfilt_size=None, ensemble_threshold=None
    ):
        tuned_decision_threshold = tune_decision_threshold(
            targets, scores, medfilt_size, ensemble_threshold
        )
        for threshold in [tuned_decision_threshold, *decision_thresholds]:
            decisions = make_decision(
                scores,
                decision_threshold=threshold,
                meanfilt_size=medfilt_size,
                ensemble_threshold=ensemble_threshold
            )
            print(
                "F@None" if threshold is None else 'F@{:.4f}:'.format(threshold),
                *['{:.4f}'.format(metric) for metric in fscore(targets, decisions)]
            )
        return tuned_decision_threshold

    for meanfilt_size in meanfilt_sizes:
        for ensemble_threshold in ensemble_thresholds:
            print(
                f'\nmeanfilt_size={meanfilt_size}, '
                f'ensemble_threshold={ensemble_threshold}'
            )
            print('NN:')
            print('Validate:')
            tuned_decision_threshold = tune_and_eval(
                nn_validate_scores, validate_targets,
                decision_thresholds=(0.5,), medfilt_size=meanfilt_size,
                ensemble_threshold=ensemble_threshold
            )
            print('Eval:')
            tune_and_eval(
                nn_eval_scores, eval_targets,
                decision_thresholds=(0.5, tuned_decision_threshold),
                medfilt_size=meanfilt_size,
                ensemble_threshold=ensemble_threshold
            )

            print('SIMPLE:')
            print('Validate:')
            tuned_decision_threshold = tune_and_eval(
                simple_validate_scores, validate_targets,
                medfilt_size=meanfilt_size,
                ensemble_threshold=ensemble_threshold
            )
            print('Eval:')
            tune_and_eval(
                simple_eval_scores, eval_targets,
                decision_thresholds=(tuned_decision_threshold,),
                medfilt_size=meanfilt_size,
                ensemble_threshold=ensemble_threshold
            )
