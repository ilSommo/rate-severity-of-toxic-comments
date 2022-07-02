__version__ = '1.0.0-rc.1'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
from torch import nn

from rate_severity_of_toxic_comments.dataset import build_dataloaders, build_dataset, load_dataframe
from rate_severity_of_toxic_comments.model import create_model
from rate_severity_of_toxic_comments.training import test_loop
from rate_severity_of_toxic_comments.utilities import parse_config, process_config
from rate_severity_of_toxic_comments.metrics import compute_metrics, plot_metrics

DEFAULT_CONFIG_FILE_PATH = 'config/default.json'
LOCAL_CONFIG_FILE_PATH = 'config/local.json'
BEST_MODELS_FILE_PATH = 'config/best_models.json'


if __name__ == '__main__':
    #TODO No need to load embedding when model is loaded from file
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--mode', default='best', choices=['best', 'last'])
    parser.add_argument('--folder', default='res/models/')
    parser.add_argument('--predictions', type=int, default=None)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    CONFIG = parse_config(DEFAULT_CONFIG_FILE_PATH, LOCAL_CONFIG_FILE_PATH)

    if args.mode == 'last':
        model_files = glob.glob(args.folder + '*.pth')
        latest_file = max(model_files, key=os.path.getctime)
        models = [{
            'description': 'Last Model Found',
            'path': latest_file
        }]
        print(f"Using weights found in {latest_file}")
    elif args.mode == 'best':
        models_file = open(BEST_MODELS_FILE_PATH)
        models = json.load(models_file)
    else:
        raise argparse.ArgumentError('Invalid mode')

    eval_dataset_params = CONFIG['evaluation']['dataset']

    batch_size = args.batch_size

    if eval_dataset_params['type'] == 'regression':
        loss_fn = nn.MSELoss()
    elif eval_dataset_params['type'] == 'ranking':
        loss_fn = nn.MarginRankingLoss(
            margin=eval_dataset_params['loss_margin'])
    else:
        loss_fn = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and CONFIG['options']['use_gpu'] else 'cpu')

    for model_details in models:
        if not os.path.isfile(model_details['path']):
            print(
                model_details['description'] +
                ' skipped since it was not found')
            continue

        if args.mode == 'best':
            run_mode, training_params, model_params = model_details['params'][
                'run_mode'], model_details['params']['training'], model_details['params']['model']
            CONFIG['options']['run_mode'] = run_mode
            CONFIG['training'].update(training_params)
            CONFIG[run_mode].update(model_params)
        else:
            run_mode = CONFIG['options']['run_mode']
            training_params = CONFIG['training']
            model_params = CONFIG[run_mode]

        df_original = pd.read_csv(eval_dataset_params["path"])
        df_test = load_dataframe(
            run_mode, eval_dataset_params, model_params=model_params)

        CONFIG['recurrent']['vocab_file'] = model_params['vocab_file']
        support_bag = process_config(df_test, CONFIG)

        test_data = build_dataset(
            df_test,
            eval_dataset_params,
            model_params,
            support_bag['tokenizer'])
        test_dl, = build_dataloaders([test_data], [batch_size], [False])

        model = create_model(
            run_mode,
            training_params,
            model_params,
            support_bag)
        model.load_state_dict(torch.load(model_details['path'], map_location=device))
        model.to(device)

        metrics = test_loop(
            test_dl,
            model,
            loss_fn,
            device,
            log_interval=1000,
            dataset_type=eval_dataset_params['type'],
            use_wandb=False,
            collect_predictions=True)
        
        y_score = [p["prediction"] for p in metrics['predictions']]
        hist = pd.DataFrame({'score': y_score})

        if not args.headless:
            plt.hist(hist, 100)
            plt.show()

        hist.to_csv(
            'res/hist/' + model_details['path'].split('/')[-1][11:-4] + '.csv')

        if eval_dataset_params['type'] == 'classification':
            y_test = [p["target"] for p in metrics['predictions']]
            eval_metrics = compute_metrics(
                torch.tensor(y_score), torch.tensor(y_test))
            print(eval_metrics)
            plot_metrics(eval_metrics)
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            roc_auc = auc(y_test, y_score)
            ns_probs = ns_probs = [0 for _ in range(len(y_test))]
            ns_auc = roc_auc_score(y_test, ns_probs)
            lr_auc = roc_auc_score(y_test, y_score)
            print('Random: ROC AUC=%.3f' % (ns_auc))
            print('Trained: ROC AUC=%.3f' % (lr_auc))
            ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
            lr_fpr, lr_tpr, _ = roc_curve(y_test, y_score)

            if not args.headless:
                plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random')
                plt.plot(lr_fpr, lr_tpr, marker='.', label='Trained')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.show()

            points = pd.DataFrame({'lr_fpr': lr_fpr, 'lr_tpr': lr_tpr})
            points.to_csv(
                'res/roc/' + model_details['path'].split('/')[-1][11:-4] + '.csv')
        else:
            print(model_details['description'], metrics)

        if args.predictions:
            for p in sorted(metrics["predictions"], key=lambda x: x["error"], reverse=True)[:args.predictions]:
                row = df_original[df_original['id'] == p['idx']]
                print(row["text"].values[0])
                print("="*100)
