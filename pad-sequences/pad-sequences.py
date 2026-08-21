import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        L = max_len if max_len is not None else 0
        return np.empty((0, L), dtype='int64')
        
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)
    
    seqs_r = []
    for seq in seqs:
        seq_list = list(seq) 
        
        if len(seq_list) > max_len:
            seq_list = seq_list[:max_len]
            
        elif len(seq_list) < max_len:
            seq_list.extend([pad_value] * (max_len - len(seq_list)))

        seqs_r.append(seq_list)
            
    return np.array(seqs_r, dtype='int64')
