import os
import shutil

def create_directory(path):
    """
    Creates a new directory at the given path, if it doesn't already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def delete_directory(path):
    """
    Deletes the directory at the given path, along with all its contents.
    """
    if os.path.exists(path):
        shutil.rmtree(path)

def count_files(path):
    """
    Counts the number of files in the directory at the given path (excluding subdirectories).
    """
    count = 0
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            count += 1
    return count

def read_file(path):
    """
    Reads the contents of a text file at the given path and returns them as a string.
    """
    with open(path, 'r') as file:
        return file.read()

def write_file(path, content):
    """
    Writes the given string to a text file at the given path.
    """
    with open(path, 'w') as file:
        file.write(content)
