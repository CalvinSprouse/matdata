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
    """
    Loads data from a MATLAB mat file using scipy.io.loadmat.

    Parameters:
    ----------
    mat_file : PathLike
        The path to the MATLAB mat file.
    variable_names : Sequence, optional
        A sequence of variable names to load from the mat file. If None, all variables are loaded.
    squeeze_me : bool, optional
        Whether to squeeze the loaded data. Default is True.
    simplify_cells : bool, optional
        Whether to simplify cell arrays. Default is True.
    **kwargs
        Additional keyword arguments to pass to scipy.io.loadmat.

    Returns:
    -------
    dict
        The data loaded from the MATLAB mat file.
    """
    return sio.loadmat(
        mat_file,
        variable_names=variable_names,
        squeeze_me=squeeze_me,
        simplify_cells=simplify_cells,
        **kwargs
    )


def load_h5py(mat_file: PathLike, variable_names: Sequence = None, squeeze_me: bool = True, simplify_cells: bool = True) -> dict:
    """
    Load MATLAB v7.3 .mat files using h5py with automatic transposition and
    conversion of 1x1 arrays to scalars.

    Parameters:
        filepath (str): Path to the .mat file.
        squeeze_me (bool): If True, squeeze unit dimensions from arrays.
        simplify_cells (bool): If True, convert MATLAB cell arrays to lists.

    Returns:
        dict: Dictionary containing MATLAB variables.
    """
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
    """
    A class to help with loading and accessing data from MATLAB mat files.

    Attributes:
    ----------
    mat_file : Path
        The path to the MATLAB mat file.
    data : dict
        The data loaded from the MATLAB mat file.

    Methods:
    -------
    __init__(self, mat_file: PathLike, variable_names: Sequence = None)
        Initializes the MatData object and loads the data from the specified mat file.

    get_file(self) -> Path
        Returns the path to the MATLAB mat file.

    get(self, var: str) -> npt.ArrayLike
        Returns the data for the specified variable from the mat file.

    get_keys(self) -> Sequence[str]
        Returns the keys of the data dictionary.

    __repr__(self)
        Returns a string representation of the MatData object, including the file location and keys.
    """

    def __init__(self, mat_file: PathLike, variable_names: Sequence = None, version: float = None):
        """
        Initializes the MatData object and loads the data from the specified mat file.

        Parameters:
        ----------
        mat_file : PathLike
            The path to the MATLAB mat file.
        variable_names : Sequence, optional
            A sequence of variable names to load from the mat file. If None, all variables are loaded.
        version : float, optional
            The version of the MATLAB file format. If specified, the data will be loaded using the appropriate method.

        Raises:
        ------
        FileNotFoundError
            If the specified mat file does not exist.
        """
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
        """
        Returns the path to the MATLAB mat file.

        Returns:
        -------
        Path
            The path to the MATLAB mat file.
        """
        return self.mat_file


    def get(self, var: str) -> npt.ArrayLike:
        """
        Returns the data for the specified variable from the mat file.

        Parameters:
        ----------
        var : str
            The name of the variable to retrieve from the mat file.

        Returns:
        -------
        npt.ArrayLike
            The data for the specified variable.

        Raises:
        ------
        KeyError
            If the specified variable is not found in the data.
        """
        if var not in self.data:
            raise KeyError(f"Variable '{var}' not found in data")

        return self.data[var]


    def get_keys(self) -> Sequence[str]:
        """
        Returns the keys of the data dictionary.

        Returns:
        -------
        Sequence[str]
            The keys of the data dictionary.
        """
        return self.data.keys()


    def __repr__(self):
        """
        Returns a string representation of the MatData object, including the file location and keys.

        Returns:
        -------
        str
            A string representation of the MatData object.
        """
        file_location = self.mat_file.resolve()
        keys = list(self.data.keys())
        repr_str = f"MatData:\n\tmat_file={file_location}\n\tkeys={keys}"
        return repr_str
