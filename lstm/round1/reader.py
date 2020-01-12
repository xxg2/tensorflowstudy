import os
from six.moves import range
from six.moves.urllib.request import urlretrieve
import tensorflow as tf

dir_name = 'datasets'
def check_file(file_name, url):
    if not os.path.exists(os.path.join(dir_name,file_name)) and url == '':
        print('文件',file_name, '不存在，请确保文件在datasets目录下。')
        return False
        #file_name = urlretrieve(url + file_name, dir_name + os.sep + file_name)
    else:
        print('文件 ',file_name, '存在。盛宴大幕拉开了。')
        return True

def read_data(file_name):
    with open(os.path.join(dir_name,file_name)) as f:
        data = tf.compat.as_str(f.read())
        # 将所有文本均设为小写形式
        data = data.lower()
        data = list(data)
    return data

def maybe_download(file_name, url):
    """如果不存在，请下载文件，并确保其大小合适."""
    if not os.path.exists(file_name):
        print('下载文件中......')
        file_name, _ = urlretrieve(url + file_name, file_name)
        statinfo = os.stat(file_name)
        print('找到并验证 %s' % file_name)
    else:
        print(statinfo.st_size)
        raise Exception(
          '无法验证 ' + file_name + '.你能使用浏览器来获取吗?')
    return file_name
#filename = maybe_download('wikipedia2text-extracted.txt.bz2')