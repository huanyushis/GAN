from easydict import EasyDict as edict
cfg = edict()

cfg.name = "data_set"
cfg.seed = 100
cfg.noise_dim = 100
cfg.epochs = 50000
cfg.batch_size = 512
cfg.learning_rate = 0.0002
cfg.is_training = True