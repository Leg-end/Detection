import tensorlib.training as training
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    flags = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
    over_flags = {'a1': 1, 'b2': 2, 'c3': 3, 'd4': 4, 'e5': 5, 'f6': 6, 'g7': 7, 'h8': 8}
    hp = training.HParams(**flags)
    # hp.override_from_dict(over_flags)
    print(hp.repr__())
    # print(hp.to_json())
    # print(hp.values())
    hp.parse("a=1.2,b=2")
    print(hp.repr__())
