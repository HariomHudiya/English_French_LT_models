import pickle

def pkl_file_saver(filepath_save,content_save):
    """
    Saves file Pickle (only)
    :param content_save: What to Save
    :param filepath_save: Where to Save
    :return:
    """
    print("Saving File..")
    with open(filepath_save,'wb') as f:
        pickle.dump(content_save,f)
    print(" Saved Successfully....")


def pkl_file_loader(filepath):
    """

    :param filepath: Extracting file path
    :return: file
    """
    print("Loading Pickle File")

    with open(filepath,'rb') as f:
        file = pickle.load(f)
    print("File Loaded Successfully")

    return file



def pkl_file_saver_2(filepath_save,content_save):
    """
    Saves file Pickle (only)
    :param content_save: What to Save
    :param filepath_save: Where to Save
    :return:
    """
    print("Saving File..")
    with open(filepath_save,'w') as f:
        pickle.dump(content_save,f)
    print(" Saved Successfully....")


def pkl_file_loader_2(filepath):
    """

    :param filepath: Extracting file path
    :return: file
    """
    print("Loading Pickle File")

    with open(filepath,'r') as f:
        file = pickle.load(f)
    print("File Loaded Successfully")

    return file
