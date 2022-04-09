import shutil
import zipfile

import requests
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
    zip_file.extractall('user_model')
    zip_file.close()
    os.rename(zip_file.filename.split('.')[0], "user_model")
    return 'user_model'


def get_file(url):
    del_all()
    if(url.startswith('http')):
        r = requests.get(url=url, stream=True)
        f = open('user_model.zip', 'wb')
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
        f.close()
        return 'user_model.zip'
    else:
        return url


def del_all():
    if(os.path.exists('user_model')):
        shutil.rmtree('user_model')
    if(os.path.exists('user_model.zip')):
        os.remove('user_model.zip')


p = unzip_file(get_file('http://localhost:8000/files/ptm-leaderboard/user1/sa/user_model00.zip?t=1649491872947076400'))
print(p)
