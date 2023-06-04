import os
import shutil
import csv
from pathlib import Path

# images = [f for f in os.listdir() if '.jpg' in f.lower()]

# os.mkdir('downloaded_images')

# for image in images:
#     new_path = 'downloaded_images/' + image
#     shutil.move(image, new_path)



def movefiles(dataset):
    file = f'./nsynth-{dataset}/{dataset}.csv'
    # Open file
    with open(file) as file_obj:
        # Create reader object by passing the file
        # object to reader method
        reader_obj = csv.reader(file_obj)

        # Iterate over each row in the csv
        # file using reader object
        for (i, row) in enumerate(reader_obj):
            old_path = f'/Users/mariechu/Downloads/{dataset}/{row[1]}'
            if os.path.exists(old_path):
                if not os.path.exists(f'{dataset}/{row[0]}'):
                    os.makedirs(f'{dataset}/{row[0]}')
                new_path = f'{dataset}/{row[0]}/{row[1]}'
                shutil.copy(old_path, new_path)

def main():
    movefiles("train")
    movefiles("test")
    movefiles("valid")

if __name__ == main():
    main()
