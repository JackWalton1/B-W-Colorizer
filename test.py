import os

def countFrames(dir_path):
    count = 0
    for entry in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, entry)):
            count += 1

    return count


# Example usage
print(countFrames('./frames/GrapesOfWrathTrailerClip/output'))