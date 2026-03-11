import gzip
import json
import os
import shutil
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xgboost as xgb
from models.basemodel import BaseModel
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset
from utils.scorer import BinScorer, ClassScorer, RegScorer
from utils.timer import Timer


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_filepath(name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD.<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return (name + "_%s." + extension) % timestr


def is_jsonable(x, cls=None):
    try:
        json.dumps(x, cls=cls)
        return True
    except (TypeError, OverflowError):
        return False


def get_scorer(objective):
    if objective == "regression":
        return RegScorer()
    elif objective == "classification":
        return ClassScorer()
    elif objective == "binary":
        return BinScorer()
    else:
        raise NotImplementedError('No scorer for "' + objective + '" implemented')


class ExperimentResult:
    """
    container class for an experiment result.

    attributes:
    - dataset(TabularDataset): a dataset object
    - scaler(str): scaler for numerical features
    - model(BaseModel): the model trained & evaluated on the dataset
    - timers(dict[Timer]): timers for training and evaluating model
    - scorers(dict): scorer objects for train, test, and val sets
    - predictions(dict): output of the model on the dataset. keys = "train", "test", "val"
    - probabilities(dict): probabilities of predicted class (only for classification problems)
    - ground_truth(dict): ground truth for each prediction, stored here just for convenience.
    - hparam_source(str): a string describing how the hyperparameters were generated
    - trial_number(int): trial number

    attributes "predictions", "probabilities", and "ground_truth" each have the same shape as the lists in dataset.split_indeces.
    """

    def __init__(
        self,
        dataset,
        scaler,
        model,
        timers,
        scorers,
        predictions,
        probabilities,
        ground_truth,
        cluster_shift=None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.model = model
        self.timers = timers
        self.scorers = scorers
        self.predictions = predictions
        self.probabilities = probabilities
        self.ground_truth = ground_truth
        self.cluster_shift = cluster_shift

        # we will set these after initialization
        self.hparam_source = None
        self.trial_number = None
        self.experiemnt_args = None
        self.exception = None

    def write(self, filepath_base, write_predictions=False, compress=False):
        """
        write two files:
        - one with the results from the trial, including metadata and performance, and
        - if self.write_predictions, write one filew with all metadata, all predictions, ground truth, and split indices.
        """

        # create a dict with all output we want to store
        result_dict = {
            "dataset": self.dataset.get_metadata(),
            "scaler": self.scaler,
            "model": self.model.get_metadata(),
            "experiemnt_args": self.experiment_args,
            "hparam_source": self.hparam_source,
            "trial_number": self.trial_number,
            "exception": str(self.exception),
            "timers": {name: timer.save_times for name, timer in self.timers.items()},
            "scorers": {
                name: scorer.get_results() for name, scorer in self.scorers.items()
            },
            "cluster_shift": self.cluster_shift,
        }

        # write results
        for k, v in result_dict.items():
            if not is_jsonable(v, cls=NpEncoder):
                raise Exception(
                    f"writing results: value at key '{k}' is not json serializable: {v}"
                )

        write_dict_to_json(
            result_dict,
            Path(str(filepath_base) + "_results.json"),
            compress=compress,
            cls=NpEncoder,
        )

        if write_predictions:
            # add the predictions (lots of data) to a new dict
            prediction_dict = result_dict.copy()

            prediction_dict["predictions"] = self.predictions
            prediction_dict["probabilities"] = self.probabilities
            prediction_dict["ground_truth"] = self.ground_truth
            prediction_dict["splits"] = [
                {key: list(val.tolist()) for key, val in split.items()}
                for split in self.dataset.split_indeces
            ]

            # write predictions
            for k, v in prediction_dict.items():
                if not is_jsonable(v, cls=NpEncoder):
                    raise Exception(
                        f"writing predictions: value at key '{k}' is not json serializable: {v}"
                    )

            write_dict_to_json(
                prediction_dict,
                Path(str(filepath_base) + "_predictions.json"),
                compress=compress,
                cls=NpEncoder,
            )


class TimeoutException(Exception):
    pass


def cross_validation(
    model: BaseModel,
    dataset: TabularDataset,
    time_limit: int,
    scaler: str,
    args: NamedTuple,
) -> ExperimentResult:
    """
    takes a BaseModel and TabularDataset as input, and trains and evaluates the model using cross validation with all
    folds specified in the dataset property split_indeces. Time limit is checked after each fold, and an exception is raised.
    Scaler is passed to tabzilla_data_processing.process_data()

    returns an ExperimentResult object, which contains all metadata and results from the cross validation run, including:
    - evlaution objects for the validation and test sets
    - predictions and prediction probabilities for all data points in each fold.
    - runtimes for training and evaluation, for each fold
    """

    # Record some statistics and metrics
    # create a scorer & timer object for the train, val, and test sets
    scorers = {
        "train": get_scorer(dataset.target_type),
        "val": get_scorer(dataset.target_type),
        "test": get_scorer(dataset.target_type),
    }
    timers = {
        "train": Timer(),
        "val": Timer(),
        "test": Timer(),
        "train-eval": Timer(),
    }

    # store predictions and class probabilities. probs will be None for regression problems.
    # these have the same dimension as train_index, test_index, and val_index
    predictions = {
        "train": [],
        "val": [],
        "test": [],
    }
    probabilities = {
        "train": [],
        "val": [],
        "test": [],
    }
    ground_truth = {
        "train": [],
        "val": [],
        "test": [],
    }

    cluster_shift = {
        "tv": [],
        "p_train": [],
        "p_test": [],
        "k": args.cluster_k if hasattr(args, "cluster_k") else None,
        "pca_dim": args.cluster_pca_dim if hasattr(args, "cluster_pca_dim") else None,
    }

    start_time = time.time()
    # iterate over all train/val/test splits in the dataset property split_indeces
    for i, split_dictionary in enumerate(dataset.split_indeces):
        if time.time() - start_time > time_limit:
            raise TimeoutException(
                f"time limit of {time_limit}s reached during fold {i}"
            )

        train_index = split_dictionary["train"]
        val_index = split_dictionary["val"]
        test_index = split_dictionary["test"]

        # run pre-processing & split data
        processed_data = process_data(
            dataset,
            train_index,
            val_index,
            test_index,
            verbose=False,
            scaler=scaler,
            one_hot_encode=False,
            args=args,
        )
        X_train, y_train = processed_data["data_train"]
        X_val, y_val = processed_data["data_val"]
        X_test, y_test = processed_data["data_test"]
        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        timers["train"].start()
        # loss history can be saved if needed
        loss_history, val_loss_history = curr_model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
        )
        timers["train"].end()

        if getattr(args, "compute_cluster_shift", False) and hasattr(curr_model, "get_leaf_embeddings"):
            leaf_train = curr_model.get_leaf_embeddings(X_train)
            # Old approach: also compute test leaf embeddings and assign test samples by nearest cluster centroid.
            # leaf_test = curr_model.get_leaf_embeddings(X_test)

            encoder = OneHotEncoder(handle_unknown="ignore")
            encoded_train = encoder.fit_transform(leaf_train)
            # encoded_test = encoder.transform(leaf_test)

            if encoded_train.shape[1] > 1:
                reduction_dim = min(args.cluster_pca_dim, encoded_train.shape[1] - 1)
                reducer = TruncatedSVD(
                    n_components=reduction_dim,
                    random_state=args.cluster_seed,
                )
                z_train = reducer.fit_transform(encoded_train)
                # z_test = reducer.transform(encoded_test)
            else:
                reduction_dim = encoded_train.shape[1]
                z_train = encoded_train.toarray()
                # z_test = encoded_test.toarray()


            # # Temporary diagnostic: save a 2D train scatter and an elbow plot for this fold.
            # max_k = min(args.cluster_k, len(z_train))
            # if max_k >= 2:
            #     import matplotlib

            #     matplotlib.use("Agg")
            #     import matplotlib.pyplot as plt

            #     elbow_dir = Path("cluster_k_sweeps") / args.model_name / args.dataset
            #     elbow_dir.mkdir(parents=True, exist_ok=True)
            #     diagnostic_stamp = time.time_ns()

            #     if z_train.shape[1] >= 2:
            #         scatter_path = elbow_dir / (
            #             f"fold_{i}_train_projection_{diagnostic_stamp}.png"
            #         )

            #         # Old temporary version: uncolored 2D projection.
            #         # fig, ax = plt.subplots(figsize=(6, 5))
            #         # ax.scatter(z_train[:, 0], z_train[:, 1], s=10, alpha=0.6)
            #         # ax.set_xlabel("Component 1")
            #         # ax.set_ylabel("Component 2")
            #         # ax.set_title(
            #         #     f"Train Projection: {args.model_name} / {args.dataset} / fold {i}"
            #         # )
            #         # ax.grid(True, alpha=0.3)
            #         # fig.tight_layout()
            #         # fig.savefig(scatter_path, dpi=150)
            #         # plt.close(fig)

            #     sweep_ks = list(range(2, max_k + 1))
            #     sweep_losses = []
            #     for sweep_k in sweep_ks:
            #         sweep_kmeans = KMeans(
            #             n_clusters=sweep_k,
            #             random_state=args.cluster_seed,
            #             n_init="auto",
            #         )
            #         sweep_kmeans.fit(z_train)
            #         sweep_losses.append(float(sweep_kmeans.inertia_))

            #     elbow_path = elbow_dir / (
            #         f"fold_{i}_kmax_{args.cluster_k}_{diagnostic_stamp}.png"
            #     )

            #     fig, ax = plt.subplots(figsize=(6, 4))
            #     ax.plot(sweep_ks, sweep_losses, marker="o")
            #     ax.set_xlabel("k")
            #     ax.set_ylabel("KMeans loss (inertia)")
            #     ax.set_title(f"Elbow Plot: {args.model_name} / {args.dataset} / fold {i}")
            #     ax.grid(True, alpha=0.3)
            #     fig.tight_layout()
            #     fig.savefig(elbow_path, dpi=150)
            #     plt.close(fig)

            kmeans = KMeans(n_clusters=args.cluster_k, random_state=args.cluster_seed, n_init="auto")
            c_train = kmeans.fit_predict(z_train)

            # if max_k >= 2 and z_train.shape[1] >= 2:
            #     fig, ax = plt.subplots(figsize=(6, 5))
            #     scatter = ax.scatter(
            #         z_train[:, 0],
            #         z_train[:, 1],
            #         c=c_train,
            #         cmap=plt.cm.get_cmap("tab20", args.cluster_k),
            #         s=10,
            #         alpha=0.7,
            #     )
            #     ax.set_xlabel("Component 1")
            #     ax.set_ylabel("Component 2")
            #     ax.set_title(
            #         f"Train Projection by Cluster: {args.model_name} / {args.dataset} / fold {i}"
            #     )
            #     ax.grid(True, alpha=0.3)
            #     cbar = fig.colorbar(scatter, ax=ax)
            #     cbar.set_label("Cluster")
            #     fig.tight_layout()
            #     fig.savefig(scatter_path, dpi=150)
            #     plt.close(fig)

            cluster_classifier_params = {
                "n_estimators": 100,
                "random_state": args.cluster_seed,
                "verbosity": 0,
            }
            if args.cluster_k > 2:
                cluster_classifier_params["objective"] = "multi:softprob"
                cluster_classifier_params["num_class"] = args.cluster_k
                cluster_classifier_params["eval_metric"] = "mlogloss"
            else:
                cluster_classifier_params["objective"] = "binary:logistic"
                cluster_classifier_params["eval_metric"] = "logloss"

            cluster_classifier = xgb.XGBClassifier(**cluster_classifier_params)
            cluster_classifier.fit(X_train, c_train)
            c_test = cluster_classifier.predict(X_test).astype(int)

            #if we wanted to train a classifier on projections of the tree leaf embeddings instead of original embeddings.
            # cluster_classifier.fit(z_train, c_train)
            # c_test = cluster_classifier.predict(z_test).astype(int)


            # Old approach: assign each test sample to the nearest training-space centroid.
            # c_test = kmeans.predict(z_test)

            k = args.cluster_k
            p_tr = np.bincount(c_train, minlength=k) / len(c_train)
            p_te = np.bincount(c_test, minlength=k) / len(c_test)

            tv = 0.5 * np.abs(p_tr - p_te).sum()

            cluster_shift["tv"].append(float(tv))
            cluster_shift["p_train"].append(p_tr.tolist())
            cluster_shift["p_test"].append(p_te.tolist())
            cluster_shift["pca_dim"] = reduction_dim

        # evaluate on train set
        timers["train-eval"].start()
        train_predictions, train_probs = curr_model.predict_wrapper(
            X_train, args.subset_rows
        )
        timers["train-eval"].end()
        # evaluate on val set
        timers["val"].start()
        val_predictions, val_probs = curr_model.predict_wrapper(X_val, args.subset_rows)
        timers["val"].end()
        # evaluate on test set
        timers["test"].start()
        test_predictions, test_probs = curr_model.predict_wrapper(
            X_test, args.subset_rows
        )
        timers["test"].end()
        extra_scorer_args = {}
        if dataset.target_type == "classification":
            extra_scorer_args["labels"] = range(dataset.num_classes)

        # evaluate on train, val, and test sets
        scorers["train"].eval(
            y_train, train_predictions, train_probs, **extra_scorer_args
        )
        scorers["val"].eval(y_val, val_predictions, val_probs, **extra_scorer_args)
        scorers["test"].eval(y_test, test_predictions, test_probs, **extra_scorer_args)

        # store predictions & ground truth

        # train
        predictions["train"].append(train_predictions.tolist())
        probabilities["train"].append(train_probs.tolist())
        ground_truth["train"].append(y_train.tolist())

        # val
        predictions["val"].append(val_predictions.tolist())
        probabilities["val"].append(val_probs.tolist())
        ground_truth["val"].append(y_val.tolist())

        # test
        predictions["test"].append(test_predictions.tolist())
        probabilities["test"].append(test_probs.tolist())
        ground_truth["test"].append(y_test.tolist())

    return ExperimentResult(
        dataset=dataset,
        scaler=scaler,
        model=model,
        timers=timers,
        scorers=scorers,
        predictions=predictions,
        probabilities=probabilities,
        ground_truth=ground_truth,
        cluster_shift=cluster_shift,
    )


def write_dict_to_json(x: dict, filepath: Path, compress=False, cls=None):
    assert not filepath.is_file(), f"file already exists: {filepath}"
    assert filepath.parent.is_dir(), f"directory does not exist: {filepath.parent}"
    if not compress:
        with filepath.open("w", encoding="UTF-8") as f:
            json.dump(x, f, cls=cls)
    else:
        with gzip.open(str(filepath) + ".gz", "wb") as f:
            f.write(json.dumps(x, cls=cls).encode("UTF-8"))


def make_archive(source, destination):
    """
    a helper function because shutil.make_archive is too confusing on its own. adapted from:
    http://www.seanbehan.com/how-to-use-python-shutil-make_archive-to-zip-up-a-directory-recursively-including-the-root-folder/
    zip the folder at "source" and write it to the file at "destination". the file type is read from arg "destination"

    example use:
    > make_archive("/source/directory", "/new/directory/archive.zip")
    """

    base = os.path.basename(destination)
    name = base.split(".")[0]
    format = base.split(".")[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move("%s.%s" % (name, format), destination)


import configargparse

# the parsers below are based on the TabSurvey parsers in utils.py


def get_experiment_parser():
    """parser for experiment arguments"""

    experiment_parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    experiment_parser.add(
        "-experiment_config",
        required=True,
        is_config_file=True,
        help="config file for arg parser",
    )
    experiment_parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="directory where experiment results will be written.",
    )
    experiment_parser.add_argument(
        "--write_predictions",
        action="store_true",
        help="write the predictions of each model to a json file.",
    )
    experiment_parser.add(
        "--use_gpu", action="store_true", help="Set to true if GPU is available"
    )
    experiment_parser.add(
        "--gpu_ids",
        type=int,
        action="append",
        help="IDs of the GPUs used when data_parallel is true",
    )
    experiment_parser.add(
        "--data_parallel",
        action="store_true",
        help="Distribute the training over multiple GPUs",
    )
    experiment_parser.add(
        "--n_random_trials",
        type=int,
        default=10,
        help="Number of trials of random hyperparameter search to run",
    )
    experiment_parser.add(
        "--hparam_seed",
        type=int,
        default=0,
        help="Random seed for generating random hyperparameters. passed to optuna RandomSampler.",
    )
    experiment_parser.add(
        "--n_opt_trials",
        type=int,
        default=10,
        help="Number of trials of hyperparameter optimization to run",
    )
    experiment_parser.add(
        "--batch_size", type=int, default=128, help="Batch size used for training"
    )
    experiment_parser.add(
        "--val_batch_size",
        type=int,
        default=128,
        help="Batch size used for training and testing",
    )
    experiment_parser.add(
        "--scale_numerical_features",
        type=str,
        choices=["None", "Quantile"],
        default="None",
        help="Specify scaler for numerical features. Applied during data processing, prior to training and inference.",
    )
    experiment_parser.add(
        "--early_stopping_rounds",
        type=int,
        default=20,
        help="Number of rounds before early stopping applies.",
    )
    experiment_parser.add(
        "--epochs", type=int, default=1000, help="Max number of epochs to train."
    )
    experiment_parser.add(
        "--logging_period",
        type=int,
        default=100,
        help="Number of iteration after which validation is printed.",
    )
    experiment_parser.add(
        "--experiment_time_limit",
        type=int,
        default=10,
        help="Time limit for experiment, in seconds.",
    )
    experiment_parser.add(
        "--trial_time_limit",
        type=int,
        default=10,
        help="Time limit for each train/test trial, in seconds.",
    )
    experiment_parser.add(
        "--subset_rows",
        type=int,
        default=-1,
        help="Number of rows to use for training and testing. -1 means use all rows.",
    )
    experiment_parser.add(
        "--subset_features",
        type=int,
        default=-1,
        help="Number of features to use for training and testing. -1 means use all features.",
    )
    experiment_parser.add(
        "--subset_rows_method",
        type=str,
        choices=["random", "first"],
        default="random",
        help="Method for selecting rows. 'random' means select randomly, 'first' means select the first rows.",
    )
    experiment_parser.add(
        "--subset_features_method",
        type=str,
        choices=["random", "first", "mutual_information"],
        default="random",
        help="Method for selecting features. 'random' means select randomly, 'first' means select the first features, 'mutual information' wraps sklearn's mutual_info_classif.",
    )
    experiment_parser.add(
        "--subset_random_seed",
        type=int,
        default=0,
        help="Random seed for subset selection.",
    )

    experiment_parser.add(
        "--compute_cluster_shift",
        action="store_true",
        help="Compute cluster mixture shift using tree leaf embeddings."
    )
    experiment_parser.add("--cluster_k", type=int, default=20)
    experiment_parser.add("--cluster_pca_dim", type=int, default=20)
    experiment_parser.add("--cluster_seed", type=int, default=0)

    return experiment_parser
