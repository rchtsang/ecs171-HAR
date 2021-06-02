import os
import requests
from zipfile import ZipFile
from shutil import copy
from glob import glob

# please run this script at the top level of the project
# directory or the environment will get messed up...

UCI_DIR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507"

# download dataset
print("Downloading WISDM Dataset...")
r = requests.get(f"{UCI_DIR_URL}/wisdm-dataset.zip")

# write dataset zip to file
print("Writing 'wisdm-dataset.zip' to file...")
with open("wisdm-dataset.zip", 'wb') as file:
    file.write(r.content)

# extract zip file
print("Extracting Contents to wisdm-dataset/ ...")
with ZipFile('wisdm-dataset.zip', 'r') as zf:
    zf.extractall()

# remove zip file
print("Removing wisdm-dataset.zip ...")
os.remove('wisdm-dataset.zip')

print("""\nMost of our models use phone accel data,
    so we will copy the relevant data to a dedicated
    data folder.\n""")

# create phone_accel/ directory
if not os.path.exists('phone_accel/'):
    print("Creating phone_accel/ directory...")
    os.makedirs('phone_accel')

# copy wisdm data to phone_accel directory
print("Copying phone/accel/ data to phone_accel/ directory...")
to_copy = glob(os.path.normpath("./wisdm-dataset/arff_files/phone/accel/*.arff"))
i = 0
print(f"{round(i / len(to_copy) * 100)}%", end='')
for filepath in to_copy:
    copy(filepath, os.path.normpath("phone_accel/"))
    i += 1
    print(f"\r{round(i / len(to_copy) * 100)}%", end='')
print()

print("\nFinished Initializing Dataset!")


