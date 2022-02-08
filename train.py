__version__ = '1.0.0-rc'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


from rate_severity_of_toxic_comments.dataset import build_dataset, load_dataframe, split_dataset
from rate_severity_of_toxic_comments.training import run_training
from rate_severity_of_toxic_comments.utilities import parse_config, process_config


DEFAULT_CONFIG_FILE_PATH = 'config/default.json'
LOCAL_CONFIG_FILE_PATH = 'config/local.json'


if __name__ == '__main__':
    CONFIG = parse_config(DEFAULT_CONFIG_FILE_PATH, LOCAL_CONFIG_FILE_PATH)

    run_mode = CONFIG['options']['run_mode']
    df = load_dataframe(
        run_mode,
        CONFIG['training']['dataset'],
        CONFIG[run_mode])

    support_bag = process_config(df, CONFIG)

    df_train, df_valid = split_dataset(
        df, CONFIG['training']['dataset']['target_col'], CONFIG['options']['seed'])

    training_data = build_dataset(df_train, CONFIG['training']['dataset'],
                                  CONFIG[run_mode], support_bag['tokenizer'])
    val_data = build_dataset(df_valid, CONFIG['training']['dataset'],
                             CONFIG[run_mode], support_bag['tokenizer'])

    model, loss_history = run_training(run_mode, training_data, val_data,
                                       CONFIG['training'], CONFIG[run_mode], support_bag, CONFIG['options']['seed'],
                                       CONFIG['options']['wandb'], CONFIG['options']['use_gpu'],
                                       verbose=True, log_interval=100)
