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

"""
This module contains constant variables used in the whole project.
"""

from pathlib import Path

from source.utils.import_from_mulms_az import MuLMS_AZ_Importer

MULMS_AZ_PATH: Path = Path.joinpath(
    Path(__file__).parent.parent.parent.absolute(), "mulms-az-codi2023"
).absolute()

# We load all constants from the MULMS-AZ repository to avoid duplicated code

mulms_az_importer: MuLMS_AZ_Importer = MuLMS_AZ_Importer(MULMS_AZ_PATH)
mulms_az_importer.load_module_from_mulms_az(
    "source/constants/constants.py", "constants", globals()
)

# General constants #

PROJECT_ROOT: Path = Path(
    __file__
).parent.parent.parent.absolute()  # Must be set after the function call above in order to point to the correct path

MULMS_PATH: Path = MULMS_AZ_PATH.joinpath("./data/mulms_corpus")
OTHER_DATASET_PATH: Path = PROJECT_ROOT.joinpath("./data")
SOFC_PATH: Path = OTHER_DATASET_PATH.joinpath("./sofc_exp_corpus")
MSPT_PATH: Path = OTHER_DATASET_PATH.joinpath("./annotated-materials-syntheses/data_xmi_annotated")
CODE_PATH: Path = PROJECT_ROOT.joinpath("./source")
MULMS_DATASET_READER_PATH: Path = MULMS_AZ_PATH.joinpath("./source/data_handling/mulms_dataset.py")
SOFC_DATASET_READER_PATH: Path = CODE_PATH.joinpath("./data_handling/sofc_dataset.py")

MULMS_NE_VOCAB_PATH: Path = PROJECT_ROOT.joinpath("./source/relation_extraction/vocabs/ne.vocab")
MULMS_RELATION_VOCAB_PATH: Path = PROJECT_ROOT.joinpath(
    "./source/relation_extraction/vocabs/rel.vocab"
)
SOFC_NE_VOCAB_PATH: Path = PROJECT_ROOT.joinpath(
    "./source/relation_extraction/vocabs/sofc_ne.vocab"
)
SOFC_RELATION_VOCAB_PATH: Path = PROJECT_ROOT.joinpath(
    "./source/relation_extraction/vocabs/sofc_rel.vocab"
)
MSPT_NE_VOCAB_PATH: Path = PROJECT_ROOT.joinpath(
    "./source/relation_extraction/vocabs/mspt_ne.vocab"
)
MSPT_RELATION_VOCAB_PATH: Path = PROJECT_ROOT.joinpath(
    "./source/relation_extraction/vocabs/mspt_rel.vocab"
)


# NER related constants #

sofc_ne_labels: list = ["MATERIAL", "EXPERIMENT", "VALUE", "DEVICE"]
sofc_ne_label2id: dict = {l: i for i, l in enumerate(sofc_ne_labels)}
sofc_id2ne_label: dict = dict([(v, k) for k, v in sofc_ne_label2id.items()])

sofc_slot_labels: list = [
    "anode_material",
    "cathode_material",
    "conductivity",
    "current_density",
    "degradation_rate",
    "device",
    "electrolyte_material",
    "experiment_evoking_word",
    "fuel_used",
    "interlayer_material",
    "open_circuit_voltage",
    "power_density",
    "resistance",
    "support_material",
    "thickness",
    "time_of_operation",
    "voltage",
    "working_temperature",
]

sofc_slot_label2id: dict = {l: i for i, l in enumerate(sofc_slot_labels)}
sofc_id2slot_label: dict = dict([(v, k) for k, v in sofc_slot_label2id.items()])

mspt_ne_labels: list = [
    "Meta",
    "Property_Misc",
    "Synthesis_Apparatus",
    "Operation",
    "Property_Unit",
    "Amount_Misc",
    "Number",
    "Amount_Unit",
    "Reference",
    "Property_Type",
    "Material",
    "Material_Descriptor",
    "Characterization_Apparatus",
    "Apparatus_Unit",
    "Apparatus_Descriptor",
    "Apparatus_Property_Type",
    "Condition_Misc",
    "Condition_Unit",
    "Nonrecipe_Material",
    "Condition_Type",
    "Brand",
]
mspt_ne_label2id: dict = {l: i for i, l in enumerate(mspt_ne_labels)}
mspt_id2ne_label: dict = dict([v, k] for k, v in mspt_ne_label2id.items())


mspt_rel_labels: list = [
    "Next_Operation",
    "Recipe_Target",
    "Participant_Material",
    "Condition_Of",
    "Property_Of",
    "Descriptor_Of",
    "Apparatus_Of",
    "Coref_Of",
    "Number_Of",
    "Brand_Of",
    "Apparatus_Attr_Of",
    "Type_Of",
    "Atmospheric_Material",
    "Solvent_Material",
    "Recipe_Precursor",
    "Amount_Of",
]

mspt_rel_label2id: dict = {l: i for i, l in enumerate(mspt_rel_labels)}
mspt_id2rel_label: dict = dict([(v, k) for k, v in mspt_rel_label2id.items()])
