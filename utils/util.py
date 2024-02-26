import os


def int_to_str_with_leading_zeros(num):
 """ Converts an integer to a string with leading zeros, so that the string is always of length 4.
    Needed in colorizeVideo.py """
 return f"{num:04d}"

def countFrames(dir_path):
    count = 0
    for entry in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, entry)):
            count += 1

    return count
