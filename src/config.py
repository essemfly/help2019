class LocalConfig:
    env = 'localhost'
    TRAIN_DIR = 'sample'
    VOLUME_DIR = 'src/volume'
    LOG_DIR = 'src/logs'


class ProdConfig:
    env = 'localhost'
    TRAIN_DIR = '/data/train'
    VOLUME_DIR = '/data/volume'
    LOG_DIR = '/data/volume/logs'
