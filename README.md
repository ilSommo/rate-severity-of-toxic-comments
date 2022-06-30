# Rate severity of toxic comments

## Setup

In order to download and install all necessary dependencies run:

```console
$ pipenv install
```

In order to avoid time consuming download while running the project we suggest to download them using the [download_resources.py] script

```console
$ python download_resources.py --datasets --vocabs --models
```

## Training

The download resources script will download the best trained models.

The following command will train again the model using the <b>config/default.json</b> configuration depending on the "run-mode" (recurrent, transformer, debug)

```console
$ python train.py
```

## Evaluation

The following command will evaluate the best model (referred inside <b>config/best_models.json</b>) and evaulates them on the test set specified in the params section

```console
$ python eval.py --mode=best
```

## Documentation

To browse documentation, open `docs/index.html` with a web browser.

## Remove

```console
$ make rm
```
