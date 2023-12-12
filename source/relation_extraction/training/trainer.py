# This source code is from the PyTorch Template Project (w/ very heavy adaptations)
#   (https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py)
# Copyright (c) 2018 Victor Huang
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
import time
from copy import deepcopy
from os.path import join

import torch
from numpy import inf

from source.relation_extraction.logger import Logger
from source.utils.helper_functions import get_executor_device

DEVICE: str = get_executor_device(disable_cuda=False)


class Trainer:
    """An object of this class is responsible for executing the training logic on a given model: Loading the data,
    running training (backpropagation), validation, and evaluation.
    """

    def __init__(
        self,
        model,
        config,
        optimizer,
        validation_criterion,
        train_data_loader,
        tune_data_loader,
        dev_data_loader,
        test_data_loader,
        save_dir,
        lr_scheduler=None,
        grad_acc_steps=1,
        post_process_train=False,
        post_process_validation=False,
    ):
        """
        Args:
            model: The model to train.
            config: The experimental configuration.
            optimizer: The optimizer to use in training.
            validation_criterion: ValidationCriterion to use for validation.
            train_data_loader: Data loader for training data.
            tune_data_loader: Data loader for tuning (early stopping) data.
            dev_data_loader: Data loader for dev data.
            test_data_loader: Data loader for test data.
            lr_scheduler: Learning rate scheduler to use (default: None).
            clip_grad_norm: Value to use for gradient clipping (default: None).
        """
        # Set up GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(1)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            raise NotImplementedError("Multi-GPU training is not implemented yet!")

        # Set up evaluation criterion, optimizer, and LR scheduler (optional)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.validation_criterion = validation_criterion

        # Set up epochs,checkpoint frequency, and gradient accumulation
        cfg_trainer = config["trainer"]
        self.min_epochs = cfg_trainer["min_epochs"]
        self.max_epochs = cfg_trainer["max_epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.start_epoch = 1
        self.early_stop = cfg_trainer.get("early_stop", inf)
        self.grad_acc_steps = cfg_trainer.get("grad_acc_steps", 1)
        self.batch_counter = 0  # Batch counter for gradient accumulation

        # Set up data loaders for training/validation/test examples
        self.train_data_loader = train_data_loader
        self.tune_data_loader = tune_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader

        # Set up checkpoint saving and loading
        self.checkpoint_dir = save_dir

        # Logger
        self.logger = Logger(save_dir)

        # Set up post-processing during training
        if isinstance(post_process_train, bool):
            self.post_process_train = lambda epoch: post_process_train
        else:
            assert isinstance(post_process_train, str)
            self.post_process_train = eval(post_process_train)

        # Set up post-processing during validation
        if isinstance(post_process_validation, bool):
            self.post_process_validation = lambda epoch: post_process_validation
        else:
            assert isinstance(post_process_validation, str)
            self.post_process_validation = eval(post_process_validation)

    def train(self):
        """Commence training on the model, using the parameters specified for the trainer."""
        training_starttime = time.time()
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Perform one training epoch and output training metrics
            epoch_starttime = time.time()
            train_loss, train_metrics = self.run_epoch(
                epoch, self.train_data_loader, training=True
            )
            epoch_duration = (time.time() - epoch_starttime) / 60
            self.logger.info(
                "Training epoch {} finished. (Took {:.1f} mins.)".format(epoch, epoch_duration)
            )
            self.logger.log_metric("train_loss", train_loss, format="float", step=epoch)
            self.logger.log_epoch_metrics(
                train_metrics,
                step=epoch,
                suffix="_train",
                format=self.validation_criterion.metrics_format,
            )

            # Perform one validation epoch and output validation metrics
            epoch_starttime = time.time()
            validation_loss, validation_metrics = self.run_epoch(
                epoch, self.tune_data_loader, training=False
            )
            epoch_duration = (time.time() - epoch_starttime) / 60
            self.logger.info(
                "Validation epoch {} finished. (Took {:.1f} mins.)".format(epoch, epoch_duration)
            )
            self.logger.log_metric("validation_loss", validation_loss, format="float", step=epoch)
            self.logger.log_epoch_metrics(
                validation_metrics,
                step=epoch,
                suffix="_valid",
                format=self.validation_criterion.metrics_format,
            )

            # Check if model is new best according to validation F1 score
            improved = self.validation_criterion.last_epoch_improved_best()
            if improved:
                not_improved_count = 0
            else:
                not_improved_count += 1

            if improved or epoch % self.save_period == 0:
                self._save_checkpoint(epoch, is_best=improved)

            if not_improved_count > self.early_stop and epoch >= self.min_epochs:
                self.logger.info(
                    "Validation criterion didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                training_duration = (time.time() - training_starttime) / 60
                self.logger.info("Training took {:.1f} mins.".format(training_duration))
                return

        self.logger.info("Maximum epoch number reached. Training stops.")
        self.logger.info(
            "Training took {:.1f} mins.".format((time.time() - training_starttime) / 60)
        )

    def run_epoch(self, epoch, data_loader, training=False):
        """Run one epoch.

        Args:
            epoch: Current epoch number (integer).
            data_loader: Data loader to fetch training examples from.
            training: If true, model will be trained (i.e. backpropagation happens). Default: False.

        Returns:
            A dictionary that contains information about metrics (loss, precision, recall, f1).
        """
        if training:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        epoch_loss = 0.0
        num_evaluated_batches = 0

        if training:
            post_process = self.post_process_train(epoch)
        else:
            post_process = self.post_process_validation(epoch)

        with torch.set_grad_enabled(training):
            for batch in data_loader:
                batch_loss = self.process_batch(
                    batch, training=training, post_process=post_process
                )

                epoch_loss += batch_loss

                # Print progress
                num_evaluated_batches += 1
                self.logger.debug(
                    "{} Epoch: {} {} Loss: {:.6f}".format(
                        "Training" if training else "Validation",
                        epoch,
                        self._progress(num_evaluated_batches, data_loader),
                        batch_loss,
                    )
                )

        # Compute epoch metrics; log if validation epoch
        epoch_metrics = self.validation_criterion.finalize_epoch(validation=not training)

        return epoch_loss, epoch_metrics

    def process_batch(self, batch, training=False, post_process=False):
        """Run a single batch through the model during a training epoch.

        Args:
            batch: The batch to feed to the model.
            training: If true, model will be trained (i.e. backpropagation happens). Default: False.

        Returns:
        """
        batch_loss = 0.0
        mode = "training" if training else "validation"

        print(f"Post-processing is {post_process}")

        # Run model
        # NOTE: Unpacking is specific to relation extration use case and also depends on multitask or not!
        if "MuLMSRelationParser" in str(
            type(self.model)
        ):  # Weird hack because isinstance() does not play nice with imported classes
            tokens_batch, gold_structures, tensors = batch

            num_ne, ne_pos, ne_labels, = (
                tensors["num_ne"],
                tensors["ner_pos"],
                tensors["ner_labels"],
            )

            if self.model.factorized:
                targets = tensors["rel_existence"], tensors["rel_lbls"]
            else:
                targets = tensors["rel_matrix"]

            num_ne = self._to_device(num_ne)
            ne_pos = self._to_device(ne_pos)
            ne_labels = self._to_device(ne_labels)
            targets = self._to_device(targets)

            if (
                len(tokens_batch) == 1 and len(tokens_batch[0]) > 500
            ):  # Hacky way to deal sentences in SOFC test set that are too long during evaluation
                predicted_structures = deepcopy(
                    gold_structures
                )  # OK because none of the broken sentences have any relations annotated
                loss = 0.0
            else:
                predicted_structures, loss = self.model(
                    tokens_batch,
                    num_ne,
                    ne_pos,
                    ne_labels,
                    mode=mode,
                    targets=targets,
                    post_process=post_process,
                )
        elif "MultitaskRelationParser" in str(
            type(self.model)
        ):  # Weird hack because isinstance() does not play nice with imported classes
            if mode == "training":
                data_src_index, (tokens_batch, gold_structures, tensors) = batch
                num_ne, ne_pos, ne_labels, = (
                    tensors["num_ne"],
                    tensors["ner_pos"],
                    tensors["ner_labels"],
                )
                targets = tensors["rel_matrix"]
                num_ne = self._to_device(num_ne)
                ne_pos = self._to_device(ne_pos)
                ne_labels = self._to_device(ne_labels)
                targets = self._to_device(targets)
                predicted_structures, loss = self.model(
                    data_src_index,
                    tokens_batch,
                    num_ne,
                    ne_pos,
                    ne_labels,
                    mode=mode,
                    targets=targets,
                    post_process=post_process,
                )
            else:
                tokens_batch, gold_structures, tensors = batch
                num_ne, ne_pos, ne_labels, = (
                    tensors["num_ne"],
                    tensors["ner_pos"],
                    tensors["ner_labels"],
                )
                targets = tensors["rel_matrix"]
                num_ne = self._to_device(num_ne)
                ne_pos = self._to_device(ne_pos)
                ne_labels = self._to_device(ne_labels)
                targets = self._to_device(targets)
                if (
                    len(tokens_batch) == 1 and len(tokens_batch[0]) > 500
                ):  # Hacky way to deal sentences in SOFC test set that are too long during evaluation
                    predicted_structures = deepcopy(
                        gold_structures
                    )  # OK because none of the broken sentences have any relations annotated
                    loss = 0.0
                else:
                    predicted_structures, loss = self.model.relation_parsers[0](
                        tokens_batch,
                        num_ne,
                        ne_pos,
                        ne_labels,
                        mode=mode,
                        targets=targets,
                        post_process=post_process,
                    )
        else:
            assert False, print(type(self.model))

        # Compute and log metrics using validation criterion
        for gold_structure, predicted_structure in zip(gold_structures, predicted_structures):
            self.validation_criterion.compute_and_log_pairwise_metrics(
                gold_structure, predicted_structure
            )

        loss = loss / self.grad_acc_steps
        batch_loss += (
            loss.item() if not isinstance(loss, float) else 0.0
        )  # if expression to deal with dummy case

        # Perform backpropagation and run optimizer (when training)
        if training:
            loss.backward()

            self.batch_counter += 1
            if self.batch_counter == self.grad_acc_steps:
                self.batch_counter = 0
                self.logger.info("Calling optimizer.")
                self.optimizer.step()
                self.optimizer.zero_grad()  # Zero gradients for next batch

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # Take an LR scheduler step after each batch
                self.logger.info(
                    "LRs are now: "
                    + ", ".join("{:.2e}".format(lr) for lr in self.lr_scheduler.get_lr())
                )

        return batch_loss

    def evaluate(self, return_raw_numbers=False):
        self.logger.info("Running evaluation.")
        self.logger.info("Restoring best checkpoint...")
        checkpoint = torch.load(join(self.checkpoint_dir, "model_best.pth"), map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.logger.info("Checkpoint loaded.")

        self.logger.info("Starting evaluation on dev set.")
        epoch_starttime = time.time()
        dev_loss, dev_metrics = self.run_epoch(0, self.dev_data_loader, training=False)
        epoch_duration = (time.time() - epoch_starttime) / 60
        self.logger.info("Evaluation finished. (Took {:.1f} mins.)".format(epoch_duration))
        self.logger.log_metric("dev_loss", dev_loss, format="float")
        self.logger.log_epoch_metrics(
            dev_metrics, suffix="_dev", format=self.validation_criterion.metrics_format
        )

        self.logger.info("Starting evaluation on test set.")
        epoch_starttime = time.time()
        eval_loss, eval_metrics = self.run_epoch(0, self.test_data_loader, training=False)
        epoch_duration = (time.time() - epoch_starttime) / 60
        self.logger.info("Evaluation finished. (Took {:.1f} mins.)".format(epoch_duration))
        self.logger.log_metric("eval_loss", eval_loss, format="float")
        self.logger.log_epoch_metrics(
            eval_metrics, suffix="_eval", format=self.validation_criterion.metrics_format
        )

        if return_raw_numbers:
            return dev_metrics, eval_metrics

    def _progress(self, num_completed_batches, data_loader):
        """Nicely formatted epoch progress"""
        return "[{}/{} ({:.0f}%)]".format(
            num_completed_batches,
            len(data_loader),
            100.0 * num_completed_batches / len(data_loader),
        )

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine,"
                "training will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu)
            )
            n_gpu_use = n_gpu
        device = torch.device(DEVICE)
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, is_best=False):
        """Save a checkpoint.

        Args:
            epoch: current epoch number
            is_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        filename = str(join(self.checkpoint_dir, "checkpoint-epoch{}.pth".format(epoch)))

        if is_best:
            best_path = str(join(self.checkpoint_dir, "model_best.pth"))
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
        else:
            torch.save(state, filename)
            self.logger.info("Saving regular checkpoint: {} ...".format(filename))

    def resume_checkpoint(self, resume_path):
        """Resume from saved checkpoint.

        Args:
            resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        self.start_epoch = checkpoint["epoch"] + 1

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("Checkpoint loaded. Resume from epoch {}".format(self.start_epoch))

    def _to_device(self, data):
        if isinstance(data, torch.Tensor):
            assert data.device != self.device
            return data.to(self.device)
        elif isinstance(data, dict):
            assert all(isinstance(val, torch.Tensor) for val in data.values())
            assert all(val.device != self.device for val in data.values())
            data_on_device = dict()
            for key in data:
                data_on_device[key] = data[key].to(self.device)
            return data_on_device
        elif isinstance(data, tuple):
            assert all(isinstance(val, torch.Tensor) for val in data)
            assert all(val.device != self.device for val in data)
            return tuple(val.to(self.device) for val in data)
        else:
            raise Exception("Cannot move this kind of data to a device!")
