class MediaAccessMapper:
    def __init__(self, feats):
        self.feats

    def __call__(self, sample):
        for (k, feat) in self.feats:
            if k in sample:
                try:
                    fv = sample.pop(k)
                    assert isinstance(fv, dict) or isinstance(fv, list)
                    sample.update(feat.encode_example(k, fv))
                except Exception as e:
                    print(f'error in encoding {k=} {fv=} pkey={sample} {e}')
                    raise e
        return sample
