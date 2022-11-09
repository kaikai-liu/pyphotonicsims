from pathlib import Path
import os

""" General-purpose upload/download to Box.

    Requirements: Installing Box Drive
    link: https://www.box.com/resources/downloads/drive

"""

def get_box_path(folder='results/fresco', filename='test.txt'):
    """ Function to generate path to any file within the ocpi Box.

    Requires the user to have installed Box Drive: https://www.box.com/resources/downloads/drive


    Args:
        folder: Path of folders in the form 'folder1/folder2/folder3' etc. Must use '/' forward slash.
        filename: name of the file to import

    Returns:
        path (str): full path including filename.
    """

    # machine's user home path
    home = str(Path.home())

    # base path to box
    box_base_path = os.path.join(home, 'Box')

    base_path = box_base_path
    dir_list = folder.split('/')
    for folder in dir_list:
        base_path = os.path.join(base_path, folder)

    path = os.path.join(base_path, filename)

    return path