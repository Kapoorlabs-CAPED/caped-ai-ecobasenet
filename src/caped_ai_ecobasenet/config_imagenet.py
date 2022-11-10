import json
from dataclasses import dataclass


def save_json(data, fpath, **kwargs):
    with open(fpath, "w") as f:
        f.write(json.dumps(data, **kwargs))


@dataclass
class ParamsTrain:

    epochs: int
    learning_rate: float
    batch_size: int


@dataclass
class Files:

    train_file: str
    val_file: str
    test_file: str


@dataclass
class Paths:

    model_dir: str
    image_dir: str
    label_dir: str


@dataclass
class ImageNetConfig:

    paths_imagenet: Paths
    files_imagenet: Files
    params_train: ParamsTrain

    @classmethod
    def to_json(cls):

        cls.is_valid()
        config = {
            "epochs": cls.params_train.epochs,
            "learning_rate": cls.params_train.learning_rate,
            "batch_size": cls.params_train.batch_size,
            "model_dir": cls.paths_imagenet.model_dir,
            "image_dir": cls.paths_imagenet.image_dir,
            "label_dir": cls.paths_imagenet.label_dir,
            "train_file": cls.files_imagenet.train_file,
            "val_file": cls.files_imagenet.val_file,
            "test_file": cls.files_imagenet.test_file,
        }

        save_json(
            config, cls.paths_imagenet.model_dir + "/" + "parameters.json"
        )

    def is_valid(cls, return_invalid=False):
        """Check if configuration is valid.
        Returns
        -------
        bool
        Flag that indicates whether the current configuration values are valid.
        """

        ok = {}
        ok["epochs"] = isinstance(cls.params_train.epochs, int)
        ok["learning_rate"] = isinstance(cls.params_train.learning_rate, float)
        ok["batch_size"] = isinstance(cls.params_train.batch_size, int)
        ok["model_dir"] = isinstance(cls.paths_imagenet.model_dir, str)
        ok["image_dir"] = isinstance(cls.paths_imagenet.image_dir, str)
        ok["label_dir"] = isinstance(cls.paths_imagenet.label_dir, str)
        ok["train_file"] = isinstance(cls.files_imagenet.train_file, str)
        ok["val_file"] = isinstance(cls.files_imagenet.val_file, str)
        ok["test_file"] = isinstance(cls.files_imagenet.test_file, str)

        if return_invalid:
            return all(ok.values()), tuple(k for (k, v) in ok.items() if not v)
        else:
            return all(ok.values())
