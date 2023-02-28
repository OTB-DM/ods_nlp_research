import numpy as np

from deeppavlov.core.models.component import Component


class Mask(Component):
    """Takes a batch of tokens and returns the masks of corresponding length"""
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __call__(tokens_batch, **kwargs):
        batch_size = len(tokens_batch)
        max_len = max(len(utt) for utt in tokens_batch)
        mask = np.zeros([batch_size, max_len], dtype=np.float32)
        for n, utterance in enumerate(tokens_batch):
            mask[n, :len(utterance)] = 1

        return mask