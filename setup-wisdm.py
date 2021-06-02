import os
from os import normpath, exists
import requests
from zipfile import ZipFile
from shutil import copy, copytree
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
print("Extracting Contents to models/wisdm-dataset/ ...")
with ZipFile('wisdm-dataset.zip', 'r') as zf:
    zf.extractall('models')

# remove zip file
print("Removing wisdm-dataset.zip ...")
os.remove('wisdm-dataset.zip')

print("""\nMost of our models use phone accel data,
    so we will copy the relevant data to a dedicated
    data folder in models/.\n""")

# create models/phone_accel/ directory
if not exists('models/phone_accel/'):
    print("Creating models/phone_accel/ directory...")
    os.makedirs(normpath('models/phone_accel/'))

# copy wisdm data to phone_accel directory
print("Copying phone/accel/ data to models/phone_accel/ directory...")
to_copy = glob(normpath("./models/wisdm-dataset/arff_files/phone/accel/*.arff"))
i = 0
print(f"{round(i / len(to_copy) * 100)}%", end='') # print progress
for filepath in to_copy:
    copy(filepath, normpath("models/phone_accel/"))
    i += 1
    print(f"\r{round(i / len(to_copy) * 100)}%", end='') # print progress
print()

# copy phone_accel directory to the webapp if the folder exists
if exists('webapp/') \
        and not exists(normpath('webapp/models/phone_accel/')):
    print("Copying phone_accel/ to webapp/models/")
    copytree(normpath('models/phone_accel/'), 
        normpath('webapp/models/phone_accel/'))

print("\nFinished Initializing Dataset!")


