import os
import yaml


PACKAGE_PATH = os.path.join(os.path.dirname(__file__))

with open(os.path.join(PACKAGE_PATH, "config/songcn_params.yml")) as f:
    SONGCN_PARAMS = yaml.safe_load(f)

with open(os.path.join(PACKAGE_PATH, "config/songcn_default.yml")) as f:
    SONGCN_DEFAULT = yaml.safe_load(f)


def test():
    print(SONGCN_DEFAULT)
    print(SONGCN_PARAMS)


if __name__ == "__main__":
    test()
