# Experiment resources related to the MuLMS corpus (WIESP 2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from datasets import Dataset as HFDataset
from datasets import load_dataset

from source.constants.mulms_constants import (
    MULMS_DATASET_READER_PATH,
    MULMS_FILES,
    MULMS_PATH,
)


def load_mulms_hf_dataset(split: str) -> HFDataset:
    """
    Loads MuLMS and its requested split and returns the HuggingFace dataset.

    Args:
        split (str): Split of MuLMS

    Returns:
        HFDataset: HuggingFace Dataset object
    """
    return load_dataset(
        MULMS_DATASET_READER_PATH.__str__(),
        data_dir=MULMS_PATH.__str__(),
        data_files=MULMS_FILES,
        name="MuLMS_Corpus",
        split=("train" if split == "tune" else split),
    )
