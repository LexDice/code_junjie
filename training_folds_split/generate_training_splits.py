import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from compute_pns_split import read_momenta
from classification import Classification
import xml.dom.minidom
import glob
import pyvista as pv

NUM_SAMPLES = 177
N_POS_SAMPLES = 34
N_NEG_SAMPLES = 143

def read_xml_ids(input_path):
    # parse the XML file and get subject ids
    xml_doc = xml.dom.minidom.parse(input_path)
    xml_subjects = xml_doc.getElementsByTagName('subject')
    subject_ids = [subject_id.getAttribute('id') for subject_id in xml_subjects]
    return subject_ids


# read momenta
pos_sbj_ids = read_xml_ids('../pos_data_set.xml')
neg_sbj_ids = read_xml_ids('../neg_data_set.xml')
sbj_ids = np.concatenate((pos_sbj_ids, neg_sbj_ids), axis=0)

labels = np.array([1] * N_POS_SAMPLES + [0] * N_NEG_SAMPLES)

dataset_path = '/Users/junjiezhao/unc/Classes/COMP790/hippo/*/'
# indices_momenta = np.arange(NUM_SAMPLES)


for i in range(1000):
    train_sbjs, test_sbjs, y_train, y_test = Classification.partition(sbj_ids[:, np.newaxis], labels, 0.2, i)

    # mean_vertices = np.zeros(vertices.shape, dtype=np.float32)
    hasInit= False
    for sbj in train_sbjs:
        train_sbjs_path = glob.glob(os.path.join(dataset_path, f"*{sbj[0]}*.vtk"))

        mesh = pv.read(train_sbjs_path[0])
        vertices = np.array(mesh.points).astype(np.float32)
        if not hasInit:
            mean_vertices = vertices
            hasInit = True
        else:
            mean_vertices = mean_vertices + vertices

    num_subjs = float(len(train_sbjs))
    mean_vertices = mean_vertices / num_subjs

    mean_obj_mesh = pv.PolyData(mean_vertices, mesh.faces.reshape(-1, 4))
    mean_obj_mesh.save(f'./training_folders_vtks/fold{i}_mean_obj.vtk')


