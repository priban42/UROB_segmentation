from pathlib import Path
import os
import shutil
import random

ROOT = Path(__file__).parent
RGB = ROOT / 'rgb'
SEG = ROOT/'seg'
DATASET = ROOT/'dataset'
TRAIN = DATASET/'train'
TEST = DATASET/'test'

def refresh_dir(path):
    """
    Wipes a directory and all its content if it exists. Creates a new empty one.
    :param path:
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)
    os.makedirs(path)


def main():
    images = sorted(os.listdir(RGB))
    segmentations = sorted(os.listdir(SEG))
    assert len(images) == len(segmentations), f"len(images)={len(images)}, len(segmentations)={len(segmentations)}"
    random.seed(10)
    c = list(zip(images, segmentations))
    random.shuffle(c)

    refresh_dir(DATASET)
    refresh_dir(TRAIN)
    refresh_dir(TEST)
    refresh_dir(TRAIN/'rgb')
    refresh_dir(TRAIN/'seg')
    refresh_dir(TEST/'rgb')
    refresh_dir(TEST/'seg')
    for img, seg in c[:len(c)//3]:
        shutil.copyfile(RGB / img, TEST / 'rgb' / img)
        shutil.copyfile(SEG / seg, TEST/ 'seg' / seg)
    for img, seg in c[len(c)//3:]:
        shutil.copyfile(RGB / img, TRAIN / 'rgb' / img)
        shutil.copyfile(SEG / seg, TRAIN / 'seg' / seg)



if __name__ == "__main__":
    main()
