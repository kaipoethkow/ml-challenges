import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple


def get_adding_problem_data(seq_length: int, num_samples: int) -> Tuple[np.array, np.array]:
    """Generate sequence of float numbers and the binary mask for the
    adding problem.

    Args:
        seq_length (int): Sequence Length of the adding problem data
        num_samples (int): Total number of samples in the data set

    Returns:
        Tuple[np.array, np.array]:
            data: a sequence of [<number>, <mask-value>] pairs
            response: a sequence of response values (sums)
    """
    
    df = pd.DataFrame(columns = ['float_seq', 'mask_seq', 'response'], dtype=object)
    
    # populate DataFrame with random floats, binary mask values and the resulting sum
    for i in tqdm(range(num_samples), 'Create data'):
        float_seq = np.array(np.random.uniform(0, 1, seq_length))
        df.at[i, 'float_seq'] = float_seq
        mask = np.zeros(seq_length)
        add_idx = np.random.choice(seq_length, size=2, replace=False)
        mask[add_idx[0]] = 1
        mask[add_idx[1]] = 1
        df.at[i, 'mask_seq'] = mask
        df.at[i, 'response'] = float_seq[add_idx[0]] + float_seq[add_idx[1]]
    
    # extract tensorflow-digestible arrays
    response = df['response'].values.astype('float32')
    data = df[['float_seq', 'mask_seq']].apply(lambda x2: np.column_stack((x2[0], x2[1])), axis=1)
    data = np.array(list(data))
    
    return data, response
