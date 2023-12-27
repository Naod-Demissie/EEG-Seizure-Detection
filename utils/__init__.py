import numpy as np


def read_lbl_file(lbl_path):
    with open(lbl_path) as f:
        lbl_lines = [line for line in f.readlines() if 'label' in line]

    lbl_lines = [lbl_line.split('{')[-1].split(']}\n')[0].split('[') for lbl_line in lbl_lines]
    time_idxs = [list(map(lambda x: float(x), lbl[0].split(', ')[2:-2])) for lbl in lbl_lines]
    chl_idxs = [int(lbl[0].split(', ')[-2]) for lbl in lbl_lines]
    cat_lbls = [list(map(lambda x: float(x), lbl[1].split(', '))) for lbl in lbl_lines]
    binary_lbls = [{6: 'bckg', 7: 'seiz'}[np.argmax(cat_lbl)] for cat_lbl in cat_lbls]

    lbl_annotations = [[chl_idx, time_idx, bi_lbl] for (chl_idx, time_idx, bi_lbl) in zip(chl_idxs, time_idxs, binary_lbls)]

    # return lbl_annotations
    lbl_dict = {}
    for lbl_annotation in lbl_annotations:
        chl_idx, time_idx, binary_lbl = lbl_annotation
        if chl_idx not in lbl_dict:
            lbl_dict[chl_idx] = []
        lbl_dict[chl_idx].append([time_idx, binary_lbl])
    return lbl_dict


# def read_lbl_file(lbl_path):
#     with open(lbl_path) as f:
#         lbl_lines = [line for line in f.readlines() if 'label' in line]

#     lbl_lines = [lbl_line.split('{')[-1].split(']}\n')[0].split('[') for lbl_line in lbl_lines]
#     time_idxs = [list(map(lambda x: float(x), lbl[0].split(', ')[:-1])) for lbl in lbl_lines]
#     cat_lbls = [list(map(lambda x: float(x), lbl[1].split(', '))) for lbl in lbl_lines]
#     binary_lbls = [{6: 'bckg', 7: 'seiz'}[np.argmax(cat_lbl)] for cat_lbl in cat_lbls]

#     lbl_annotations = [[*time_idx, bi_lbl] for (time_idx, bi_lbl) in zip(time_idxs, binary_lbls)]
#     return lbl_annotations