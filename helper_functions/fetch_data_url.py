import os
import tarfile
from urllib import request

def fetch_data(f_url=None, d_name=None):

    curr_path = os.getcwd()
    dir_path = os.path.join(curr_path, "datasets")

    os.makedirs(dir_path, exist_ok=True)
    tar_file_path = os.path.join(dir_path, d_name+".tgz")
    request.urlretrieve(f_url, filename=tar_file_path)

    file_tgz = tarfile.open(tar_file_path)
    file_tgz.extractall(path=dir_path)
    file_tgz.close()
    os.remove(path=dir_path+"/"+d_name+".tgz")