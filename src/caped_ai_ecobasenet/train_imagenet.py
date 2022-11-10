import hydra
from config_imagenet import ImageNetConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name="ImageNetConfig", node=ImageNetConfig)


@hydra.main(config_path="conf", config_name="config_imagenet")
def main(config: ImageNetConfig):

    # epochs = config.params_train.epochs
    # learning_rate = config.params_train.learning_rate
    # batch_size = config.params_train.batch_size

    # model_dir = config.paths_imagenet.model_dir
    # image_dir = config.paths_imagenet.image_dir
    # label_dir = config.paths_imagenet.label_dir

    config.to_json()


if __name__ == "main":

    main()