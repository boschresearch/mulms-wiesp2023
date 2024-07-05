#!/bin/bash

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

echo -e '\033[1;31mExecute this script from the directory containing this file! \033[0m'

PROJECT_ROOT=$(realpath "../../..")

export PYTHONPATH="$PYTHONPATH:${PROJECT_ROOT}:${PROJECT_ROOT}/mulms-az-codi2023"

### Name of BERT model ###

modelNameOrPath="matscibert" # Must be one of ["bert", "scibert", "matscibert"]

### Path Settings ###

outputDir="$PROJECT_ROOT/output/relation_extraction/mulms/${modelNameOrPath}"

######

mkdir -p $outputDir

cd $PROJECT_ROOT

for f in {1..5}; do

python source/relation_extraction/train.py                                              \
--config source/relation_extraction/configs/single_task/mulms/${modelNameOrPath}.json   \
--tune_fold $f                                                                          \
--save_dir $outputDir

done

python source/relation_extraction/finegrained_eval.py       \
--input_path_one    $outputDir/${modelNameOrPath}_unfact/1  \
--input_path_two    $outputDir/${modelNameOrPath}_unfact/2  \
--input_path_three  $outputDir/${modelNameOrPath}_unfact/3  \
--input_path_four   $outputDir/${modelNameOrPath}_unfact/4  \
--input_path_five   $outputDir/${modelNameOrPath}_unfact/5
