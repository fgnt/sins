import numpy as np
import torch
from padertorch.contrib.je.modules.conv import CNN2d, CNN1d
from padertorch.contrib.je.modules.hybrid import CRNN
from sins.systems.eval import fscore
from torch import nn
from torchvision.utils import make_grid


class MultinomialClassifier(CRNN):
    """
    >>> cnn = MultinomialClassifier(**{\
        'label_key': 'labels',\
        'cnn_2d': CNN2d(**{\
            'in_channels': 1,\
            'out_channels': 3*[32] + [16],\
            'kernel_size': 3\
        }),\
        'cnn_1d': CNN1d(**{\
            'in_channels': 1024,\
            'out_channels': 3*[32] + [10],\
            'kernel_size': 3\
        }),\
        'rnn': None,\
        'fcn': None,\
    })
    >>> inputs = {\
        'features': torch.zeros(4, 1, 64, 100),\
        'labels': torch.zeros(4).long(),\
        'seq_len': None\
    }
    >>> outputs = (probs, seq_len) = cnn(inputs)
    >>> probs.shape
    torch.Size([4, 100, 10])
    >>> review = cnn.review(inputs, outputs)
    """
    def __init__(
            self, cnn_2d: CNN2d, cnn_1d: CNN1d, rnn, fcn, *,
            post_rnn_pooling=None, label_key='scene', input_size=None
    ):
        super().__init__(
            cnn_2d, cnn_1d, rnn, fcn,
            post_rnn_pooling=post_rnn_pooling, input_size=input_size
        )
        self.label_key = label_key

    def forward(self, inputs):
        x = inputs["features"]
        seq_len = inputs["seq_len"]
        y, seq_len = super().forward(x, seq_len)
        return y, seq_len

    def review(self, inputs, outputs):
        # compute loss
        logits, seq_len = outputs
        labels = inputs[self.label_key]
        assert logits.dim() == 2, logits.shape
        assert labels.dim() == 1, labels.shape
        ce = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        # create feature snapshot
        features = inputs["features"][:3]
        if features.dim() == 4:
            features = features[:, 0]

        # create review including metrics and visualizations
        review = dict(
            loss=ce.mean(),
            scalars=dict(
                predictions=predictions,
                labels=labels
            ),
            histograms=dict(),
            images=dict(
                features=features
            )
        )
        return review

    def modify_summary(self, summary):
        if 'predictions' in summary['scalars']:
            labels = summary['scalars'].pop('labels')
            predictions = summary['scalars'].pop('predictions')
            summary['scalars']['accuracy'] = np.mean(np.array(predictions) == np.array(labels))
            n_classes = int(max(labels + predictions)) + 1
            target_mat = np.zeros((len(labels), n_classes))
            target_mat[np.arange(len(labels)), labels] = 1
            pred_mat = np.zeros((len(predictions), n_classes))
            pred_mat[np.arange(len(predictions)), predictions] = 1
            f, p, r = fscore(target_mat, pred_mat, event_wise=True)
            summary['scalars']['mean_precision'] = p.mean()
            summary['scalars']['mean_recall'] = r.mean()
            summary['scalars']['mean_fscore'] = f.mean()
            summary['scalars']['min_fscore'] = f.min()
            summary['scalars']['min_idx'] = np.argmin(f)
            summary['scalars']['max_fscore'] = f.max()
            summary['scalars']['max_idx'] = np.argmax(f)
        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)
        for key, image in summary['images'].items():
            summary['images'][key] = make_grid(
                image.unsqueeze(1).flip(2),
                normalize=True, scale_each=False, nrow=1
            )
        return summary
