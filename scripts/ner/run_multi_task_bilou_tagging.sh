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

PROJECT_ROOT=$(realpath "../..")

### Training Hyperparameters (add additional if needed) ###

lr=1e-4 # Learning Rate of BERT + Linear
lrCRF=7e-3 # Learning Rate of CRF
batchSize=32
numEpochs=100
bertModel="m3rg-iitd/matscibert"
mainMTdataset="MULMS" # <- Target Dataset used for Evaluation; must be one of ["MULMS", "SOFC", "MSPT"]
addMTdataset="SOFC" # <- Additional Dataset; must be one of ["MULMS", "SOFC", "MSPT", "ALL"]

######

### Path Settings ###

outputDir="$PROJECT_ROOT/output/ner/multi_task_bilou_tagging"

######

mkdir -p $outputDir

cd $PROJECT_ROOT/source/ner/approaches/

seed="$(shuf -i 1-100000 -n 1)" # Random Seed

for f in {1..5}; do

mkdir -p $outputDir/cv_$f

python ner_bilou_tagging.py         \
--model_name $bertModel             \
--batch_size $batchSize             \
--output_path $outputDir            \
--lr $lr                            \
--lr_crf $lrCRF                     \
$enableLrDecay                      \
--cv $f                             \
--seed $seed                        \
--num_epochs $numEpochs             \
--enable_multi_tasking              \
--main_mt_dataset $mainMTdataset    \
--add_mt_dataset $addMTdataset

done

echo "Evaluating on dev"

python ../evaluation/aggregate_cv_scores.py --input_path $outputDir --set dev

echo "Evaluating on test"

python ../evaluation/aggregate_cv_scores.py --input_path $outputDir --set test
