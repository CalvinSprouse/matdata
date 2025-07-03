# matdata.py

# imports
from os import PathLike
from pathlib import Path
from typing import Sequence
import h5py
import numpy as np
import numpy.typing as npt
import scipy.io as sio


def load_scipy(mat_file: PathLike, variable_names: Sequence = None, squeeze_me: bool = True, simplify_cells: bool = True, **kwargs) -> dict:
    return sio.loadmat(
        mat_file,
        variable_names=variable_names,
        squeeze_me=squeeze_me,
        simplify_cells=simplify_cells,
        **kwargs
    )


def load_h5py(mat_file: PathLike, variable_names: Sequence = None, squeeze_me: bool = True, simplify_cells: bool = True) -> dict:
    def _mat_to_dict(mat_obj):
        """Recursively convert MATLAB objects to Python dictionaries/lists."""
        if isinstance(mat_obj, h5py.Dataset):
            # Convert datasets to numpy arrays
            data = mat_obj[()]

            # transpose multi-dimensional arrays
            if data.ndim > 1:
                data = data.T

            # mimic squeeze_me=True
            if squeeze_me:
                data = np.squeeze(data)

            # convert single-element arrays to scalars
            if data.shape == ():
                data = data.item()

            return data

        if isinstance(mat_obj, h5py.Group):
            # convert groups to dictionaries
            return {key: _mat_to_dict(mat_obj[key]) for key in mat_obj.keys()}

        raise TypeError(f"Unsupported MATLAB object type: {type(mat_obj)}")

    def _simplify(value):
        """Simplify MATLAB cells to Python lists."""
        if isinstance(value, np.ndarray) and value.dtype.kind == 'O':
            return [_simplify(item) for item in value]

        if isinstance(value, dict):
            return {k: _simplify(v) for k, v in value.items()}

        return value

    with h5py.File(mat_file, 'r') as mat:
        # load the data as a dictionary
        data = {k: _mat_to_dict(v) for k, v in mat.items()}

        # simplify cells to lists
        if simplify_cells:
            data = _simplify(data)

        # intersect the dictionary keys with the variable_names
        if variable_names:
            variables = set(data.keys()) & set(variable_names)
            data = {var: data[var] for var in variables}

        return data


class MatData:

    def __init__(self, mat_file: PathLike, variable_names: Sequence = None, version: float = None):
        self.mat_file = Path(mat_file)
        if not mat_file.exists():
            raise FileNotFoundError(f"File '{mat_file}' not found")

        # set the load method based on the reported version, if reported
        load_method = None
        if version:
            load_method = load_scipy if version <= 7.2 else load_h5py

        # attempt to load mat file
        if load_method:
            # load mat file based on reported version
            try:
                self.data = load_method(mat_file, variable_names)
            except Exception as e:
                raise Exception(f"Error loading mat file: {e}") from e
        else:
            # attempt to load as h5py file
            # fallback to scipy loadmat
            try:
                self.data = load_h5py(mat_file, variable_names)
            except OSError:
                try:
                    self.data = load_scipy(mat_file, variable_names)
                except Exception as e:
                    raise Exception(f"Error loading mat file: {e}") from e


    def get_file(self) -> Path:
        return self.mat_file


    def get(self, var: str) -> npt.ArrayLike:
        if var not in self.data:
            raise KeyError(f"Variable '{var}' not found in data")

        return self.data[var]


    def get_keys(self) -> Sequence[str]:
        return self.data.keys()


    def __repr__(self):
        file_location = self.mat_file.resolve()
        keys = list(self.data.keys())
        repr_str = f"MatData:\n\tmat_file={file_location}\n\tkeys={keys}"
        return repr_str
