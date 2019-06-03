

#generates splits
import pandas as pd
import json as json
import shutil
from sklearn.model_selection import train_test_split

work_dir ='/Users/nfs/Desktop/morphology/'

folders = ['Annotated_image_data_set_one_0-426/', 'Annotated_image_data_set_two_427_1003/']

labels = []
filepaths = []

for f in folders:
    data_csv = pd.read_csv(work_dir + f + "via_region_data.csv")
    data_csv = data_csv[data_csv['region_count'] == 1]
    data_csv = data_csv[data_csv['region_attributes'] != '{}']
    labels_curr = [json.loads(s) for s in data_csv['region_attributes'].tolist()]
    labels_curr = [d['Normal/Abnormal'] for d in labels_curr if len(d)>0]
    labels += labels_curr

    filenames = data_csv['#filename'].tolist()

    for file in filenames:
        # print(file)
        filepath = work_dir + f + file
        filepaths.append(filepath)

print("fp ",len(filepaths))
print("lables ",len(labels))
filepaths_train, filepaths_test, labels_train, labels_test = train_test_split(filepaths, labels, test_size=0.2, random_state=42)
# print("y_val: ", y_val)
# print("y_test: ", y_test)

for filep, label in zip(filepaths_train, labels_train):
        if(label == "N"):
            #write to normal
            shutil.copy(filep, work_dir + 'data_sperm/train/normal')
        elif(label == "A"):
            shutil.copy(filep, work_dir + 'data_sperm/train/abnormal')

for filep, label in zip(filepaths_test, labels_test):
        if(label == "N"):
            #write to normal
            shutil.copy(filep, work_dir + 'data_sperm/val/normal')
        elif(label == "A"):
            shutil.copy(filep, work_dir + 'data_sperm/val/abnormal')
