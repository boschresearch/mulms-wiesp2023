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
This module imports helper functions from MuLMS-AZ to avoid duplicate code.
"""

from source.constants.mulms_constants import MULMS_AZ_PATH
from source.utils.import_from_mulms_az import MuLMS_AZ_Importer

mulms_function_importer: MuLMS_AZ_Importer = MuLMS_AZ_Importer(MULMS_AZ_PATH)
mulms_function_importer.load_module_from_mulms_az(
    "source/utils/helper_functions.py", "helper_functions", globals()
)
