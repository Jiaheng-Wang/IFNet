import random
import math
import torch



class RandomErasing:
    """
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased time duration wrt input EEG trials.
         max_area: Maximum percentage of erased time duration wrt input EEG trials.
    """

    def __init__(self, probability=0.5, min_area=0.02, max_area=1/3, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.device = device

    def _erase(self, x, chan, width):
        if random.random() > self.probability:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x[..., left:left + w].zero_()

    def __call__(self, input, *args):
        if len(input.size()) == 2:
            self._erase(input, *input.size())
        else:
            batch_size, chan, width = input.size()
            for i in range(batch_size):
                self._erase(input[i], chan, width)
        return input