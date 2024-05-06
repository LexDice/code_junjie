from xml.dom import minidom
import os
import glob

NUM_FOLDS = 1000
dataset_path = '/home/junjiez1/projects/classes/COMP790/code/new_code/training_folds_split/training_folders_vtks'
# dataset_path = '/Users/junjiezhao/unc/Classes/COMP790/hippo/neg_vtk/*.vtk'

num_splits = 8

for split in range(num_splits):
    root = minidom.Document()
    xml = root.createElement('data-set')
    root.appendChild(xml)

    i_start = int(split * NUM_FOLDS / num_splits)
    i_end = i_start + int(NUM_FOLDS / num_splits)

    for i in range(i_start, i_end):
        subject_fn = f"fold{i}_mean_obj.vtk"
        pathname = os.path.join(dataset_path, subject_fn)

        # Append one element in xml
        subject = root.createElement('subject')
        subject.setAttribute('id', f"fold{i}")

        visit = root.createElement('visit')
        visit.setAttribute('id', 'experiment')

        filename = root.createElement('filename')
        filename.setAttribute('object_id', 'hippo')
        fn_text = root.createTextNode(pathname)
        filename.appendChild(fn_text)

        visit.appendChild(filename)
        subject.appendChild(visit)
        xml.appendChild(subject)

    xml_str = root.toprettyxml(indent="\t")

    save_path_file = f"./tr_folds_dataset_split{split}.xml"

    with open(save_path_file, "w") as f:
        f.write(xml_str)
