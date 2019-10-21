import numpy as np
import torch
from einops import rearrange
from sins.systems.eval import fscore
from sins.systems.modules import CNN1d, CNN2d, AutoPool
from torch import nn
from torchvision.utils import make_grid


class BinomialClassifier(nn.Module):
    """

    >>> cnn = BinomialClassifier(**{\
        'label_key': 'labels',\
        'cnn_2d': CNN2d(**{\
            'in_channels': 1,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        }),\
        'cnn_1d': CNN1d(**{\
            'in_channels': 1024,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 10,\
            'kernel_size': 3\
        }),\
        'pooling': AutoPool(**{'n_classes': 10})\
    })
    >>> inputs = {\
        'features': torch.zeros(4, 1, 64, 100),\
        'labels': torch.zeros(4, 10),\
        'seq_len': None\
    }
    >>> outputs = (probs, seq_len) = cnn(inputs)
    >>> probs.shape
    torch.Size([4, 10, 100])
    >>> review = cnn.review(inputs, outputs)
    """
    def __init__(
            self, cnn_2d: CNN2d, cnn_1d: CNN1d, pooling, *,
            label_key='presence', decision_boundary=0.5, recall_weight=1.
    ):
        super().__init__()
        self._cnn_2d = cnn_2d
        self._cnn_1d = cnn_1d
        self.pooling = pooling
        self.label_key = label_key
        self.decision_boundary = decision_boundary
        self.recall_weight = recall_weight

    def cnn_2d(self, x, seq_len=None):
        if self._cnn_2d is not None:
            x = self._cnn_2d(x)
            if seq_len is not None:
                in_shape = [(128, n) for n in seq_len]
                out_shape = self._cnn_2d.get_out_shape(in_shape)
                seq_len = [s[-1] for s in out_shape]

        if x.dim() != 3:
            assert x.dim() == 4
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x, seq_len

    def cnn_1d(self, x, seq_len=None):
        if self._cnn_1d is not None:
            x = self._cnn_1d(x)
            if seq_len is not None:
                seq_len = self._cnn_1d.get_out_shape(seq_len)
        return x, seq_len

    def forward(self, inputs):
        x = inputs["features"]
        seq_len = inputs["seq_len"]
        x, seq_len = self.cnn_2d(x, seq_len)
        y, seq_len = self.cnn_1d(x, seq_len)
        y = 1e-3 + (1. - 2e-3) * nn.Sigmoid()(y)
        return y, seq_len

    def review(self, inputs, outputs):
        # compute loss
        scores, seq_len = outputs
        if self.pooling is not None:
            scores = self.pooling(scores, seq_len)
        else:
            scores = scores
        targets = inputs[self.label_key]
        if scores.dim() > 2 and targets.dim() == 2:
            scores, targets = torch.broadcast_tensors(scores, targets.unsqueeze(-1))
        assert targets.dim() == scores.dim()
        loss = -(
                self.recall_weight * targets * torch.log(scores)
                + (1. - targets) * torch.log(1. - scores)
        ).sum(-1)
        # loss = nn.BCELoss(reduction='none')(scores, targets).sum(-1)

        # create feature snapshot
        features = inputs["features"][:3]
        if features.dim() == 4:
            features = features[:, 0]

        # create review including metrics and visualizations
        review = dict(
            loss=loss.mean(),
            scalars=dict(
                scores=scores,
                targets=targets
            ),
            histograms=dict(),
            images=dict(
                features=features
            )
        )
        if isinstance(self.pooling, AutoPool):
            alpha = self.pooling.alpha
            if torch.is_tensor(alpha) and alpha.shape[0] == 1:
                alpha = alpha.item()
            if isinstance(alpha, float):
                review['scalars']['alpha'] = alpha
        return review

    def modify_summary(self, summary):
        if 'scores' in summary['scalars']:
            scores = np.array(summary['scalars'].pop('scores')).reshape(
                (-1, self._cnn_1d.out_channels)
            )
            decision = scores > self.decision_boundary
            targets = np.array(summary['scalars'].pop('targets')).reshape(
                (-1, self._cnn_1d.out_channels)
            )
            f, p, r = fscore(targets, decision)
            summary['scalars']['precision'] = p.mean()
            summary['scalars']['recall'] = r.mean()
            summary['scalars']['fscore'] = f.mean()
        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)
        for key, image in summary['images'].items():
            summary['images'][key] = make_grid(
                image.unsqueeze(1).flip(2),  normalize=True, scale_each=False,
                nrow=1
            )
        return summary
