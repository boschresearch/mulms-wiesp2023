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
This module contains a helper class that imports classes and functions
from the MuLMS-AZ submodule repository and makes it visible to the current
namespace.
"""

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


class MuLMS_AZ_Importer:
    """
    This importer resolves module imports to the MuLMS-AZ repo and therefore helps to avoid duplicated code.
    Some functions and classes can be shared between these two repositories.
    """

    def __init__(self, mulms_az_base_path: "str|Path") -> None:
        """
        Initializes the import class.

        Args:
            mulms_az_base_path (str|Path): Relative or absolute path to the MuLMS-AZ submodule.
        """
        self._mulms_az_base_path: "str|Path" = mulms_az_base_path
        sys.path.append(self._mulms_az_base_path)

    def load_module_from_mulms_az(
        self,
        module_path: "str|Path",
        module_name: str,
        global_vars: "dict[str|Any]",
        remove_from_python_path: bool = True,
    ) -> None:
        """_summary_

        Args:
            module_path (str|Path): Path to the desired module relative to the MuLMS-AZ root path
            module_name (str): Prefix name that should be assigned to the imported module
            global_vars (dict[str|Any]): The global variables of the importing module s.t. this function can append all imports to this dict. Can be obtained by calling globals()
            remove_from_python_path (bool): If true, MuLMS-AZ path will be removed again from the Python path. Defaults to True.
        """
        module_spec = spec_from_file_location(
            name=module_name, location=os.path.join(self._mulms_az_base_path, module_path)
        )
        mulms_az_vars = module_from_spec(module_spec)
        module_spec.loader.exec_module(mulms_az_vars)

        var_names: list[str] = [x for x in mulms_az_vars.__dict__ if not x.startswith("_")]

        global_vars.update({k: getattr(mulms_az_vars, k) for k in var_names})

        if remove_from_python_path:
            sys.path = [p for p in sys.path if p != self._mulms_az_base_path]
