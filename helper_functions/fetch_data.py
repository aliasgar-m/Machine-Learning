import os
import tarfile
from urllib import request

def get_data_url(f_url=None, d_name=None):

    curr_path = os.getcwd()
    dir_path = os.path.join(curr_path, "datasets")

    os.makedirs(dir_path, exist_ok=True)
    tar_file_path = os.path.join(dir_path, d_name+".tgz")
    request.urlretrieve(f_url, filename=tar_file_path)

    file_tgz = tarfile.open(tar_file_path)
    file_tgz.extractall(path=dir_path)
    file_tgz.close()
    os.remove(path=dir_path+"/"+d_name+".tgz")

if __name__ == "__main__":
    ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    URL = ROOT + "datasets/housing/housing.tgz"
    D_NAME = "housing"

    get_data_url(f_url=URL, d_name=D_NAME)