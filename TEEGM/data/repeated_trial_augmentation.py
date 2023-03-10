import torch
from joblib import Parallel, delayed



class RepeatedTrialAugmentation:
    def __init__(self, transform = torch.nn.Identity(), m = 5):
        self.transform = transform
        self.m = m

    def __call__(self, x, y):
        if self.m == 1:
            return self.transform(x, y), y

        # todo: could be run in parallel
        # bdata = Parallel(n_jobs=self.m)(delayed(self.transform)(x, y) for _ in range(self.m))
        # data = torch.cat(bdata, dim=0)

        # Need x.clone() if transforms are in-place!
        data = torch.cat([self.transform(x.clone(), y) for _ in range(self.m)], dim=0)
        labels = torch.cat([y]*self.m, dim=0)
        return data, labels