

def idx_key_to_label(target_dict:dict, idx_list:list):
    out_dict = {}
    for key in target_dict.keys():
        out_dict[idx_list[key]] = target_dict[key]

    return out_dict