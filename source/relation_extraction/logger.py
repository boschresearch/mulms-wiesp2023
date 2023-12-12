#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import logging
import logging.config
from os.path import join

DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(message)s"},
        "datetime": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "datetime",
            "filename": "info.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "info_file_handler"]},
}


class Logger:
    """Class for logging messages, metrics, and artifacts during training."""

    def __init__(self, save_dir, verbosity=logging.DEBUG):
        """
        Args:
            save_dir: Directory to save text log to.
            verbosity: Verbosity of text logging.
        """
        for _, handler in DEFAULT_CONFIG["handlers"].items():
            if "filename" in handler and save_dir is not None:
                handler["filename"] = str(join(save_dir, handler["filename"]))

        logging.config.dictConfig(DEFAULT_CONFIG)

        self.text_logger = logging.getLogger()
        self.text_logger.setLevel(verbosity)

    def info(self, msg):
        """Log message with level INFO with the text logger."""
        self.text_logger.info(msg)

    def debug(self, msg):
        """Log message with level DEBUG with the text logger."""
        self.text_logger.debug(msg)

    def warning(self, msg):
        """Log message with level WARNING with the text logger."""
        self.text_logger.warning(msg)

    def log_metric(self, metric_name, value, format=None, step=None):
        """Log a training/evaluation metric.

        Args:
            metric_name: Name of the metric to log.
            value: Value of the metric.
            percent: Whether to log the metric as a percentage in the text log (default: True).
            step: Epoch to log the metric for.
        """
        if format == "percent":
            self.info("{}: {:.2f}%".format(metric_name, value * 100))
        elif format == "float":
            self.info("{}: {:.4}".format(metric_name, value))
        else:
            self.info("{}: {}".format(metric_name, value))

    def log_param(self, param_name, value):
        """Log a parameter."""
        raise NotImplementedError("log_param not implemented yet.")

    def log_config(self, config):
        """Log a config.

        Args:
            config: Nested dictionary of parameters.
        """
        flat_config = dict()
        _flatten_dict(config, flat_config)

        for param, value in flat_config.items():
            self.text_logger.info(f"{param} = {value}")

    def log_epoch_metrics(self, metrics, step=None, suffix="", format=None):
        """Log metrics for one epoch.

        Args:
            metrics: Metrics to log (dictionary).
            step: Epoch to log the metrics for.
            suffix: Suffix to add to metric names (e.g. "_train").
            format: How to format the metrics (dict).
        """
        if format is not None:
            assert isinstance(format, dict)

        for metric_name in metrics.keys():
            metric_format = format[metric_name] if format else None
            self.log_metric(
                metric_name + suffix, metrics[metric_name], format=metric_format, step=step
            )


def _flatten_dict(input_dict, output_dict, prefix="", delimiter="."):
    """Flatten the nested dictionary input_dict, writing to output_dict."""
    for key, value in input_dict.items():
        if isinstance(value, dict):
            _flatten_dict(
                value, output_dict, prefix=prefix + str(key) + delimiter, delimiter=delimiter
            )
        elif isinstance(value, list):
            _flatten_dict(
                {i: x for i, x in enumerate(value)},
                output_dict,
                prefix=prefix + str(key) + delimiter,
                delimiter=delimiter,
            )
        else:
            output_dict[prefix + str(key)] = value
