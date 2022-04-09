import zipfile
import sa
import os

def attack(task_id, file_url):
    model_path = unzip_file(get_file(file_url))
    res = {}
    if(task_id == 'sa'):
        res = sa.sa_attack(model_path)
    return res


def unzip_file(name):
    zip_file = zipfile.ZipFile(name)
    zip_file.extractall('./')
    zip_file.close()
    os.rename(zip_file.filename.split('.')[0], "user_model")
    return 'user_model'


def get_file(url):
    if(url.startswith('http')):
        # download file to current dir
        pass
    else:
        return url


p = unzip_file(get_file('user_model00.zip'))
print(p)
