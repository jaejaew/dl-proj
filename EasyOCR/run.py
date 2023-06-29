from easyocr.easyocr import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)

if __name__ == '__main__':
    # Using default model
    # reader = Reader(['ko'], gpu=True)

    # Using custom model
    reader = Reader(['ko'], gpu=True,
                    model_storage_directory='../custom_network',
                    user_network_directory='../custom_network',
                    recog_network='custom')

    files, count = get_files('../demo_images/hw')

    for idx, file in enumerate(files):
        filename = os.path.basename(file)

        result = reader.readtext(file)

        # ./easyocr/utils.py 733 lines
        # result[0]: bbox
        # result[1]: string
        # result[2]: confidence
        for (bbox, string, confidence) in result:
            print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))