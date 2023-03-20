import typing

import pandas as pd
import numpy as np


def df_to_context(df: pd.DataFrame) -> typing.Tuple[list, list, np.ndarray]:
    """Helper function to creating `concepts.Context` instance

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        typing.Tuple[list, list, np.ndarray]: objects, properties and bools to create Context
    """
    objects = df.index.tolist()
    properties = list(df)
    bools = df.values

    return objects, properties, bools