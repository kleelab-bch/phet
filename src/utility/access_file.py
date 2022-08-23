import os
import pickle as pkl
import sys
import traceback


def save_data(data, file_name: str, save_path: str, mode: str = 'wb', w_string: bool = False):
    try:
        file_name = os.path.join(save_path, file_name)
        with open(file=file_name, mode=mode) as fout:
            if not w_string:
                pkl.dump(data, fout)
            elif w_string:
                fout.write(data)
    except Exception as e:
        print('\t\t## The file {0:s} can not be saved'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def load_data(file_name, load_path, mode='rb'):
    try:
        file_name = os.path.join(load_path, file_name)
        with open(file_name, mode=mode) as f_in:
            if mode == "r":
                data = f_in.readlines()
            else:
                data = pkl.load(f_in)
            return data
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e
