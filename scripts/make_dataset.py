# Python program to read
# json file

import json
import csv


# Opening JSON file
train = open('./nsynth-train/examples.json')
test = open('./nsynth-test/examples.json')
valid = open('./nsynth-valid/examples.json')


train_csv = 'train.csv'
test_csv = 'test.csv'
valid_csv = 'valid.csv'

f = valid
csv_file = valid_csv

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list
i = 0
with open(csv_file, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for example in data:
        print(data[example]['instrument_family_str'])
        writer.writerow([data[example]['instrument_family_str'], data[example]['note_str'] + ".wav"])


# Closing file
f.close()
