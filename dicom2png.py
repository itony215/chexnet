import os
import cv2
import pydicom as dicom
import json
import numpy as np
import xml.etree.cElementTree as ET
from matplotlib import pyplot as plt
from os import walk
from os.path import join
import SimpleITK as sitk

dir = 'D:/TMUH/'
raw_images_path = 'D:/EE0/'
json_files_path = 'D:/EE0/'
wrong_file=[]
a=0

def read_dcm_image(path):
       ds = sitk.ReadImage(path)
       img_array = sitk.GetArrayFromImage(ds)
       return img_array[0]

def normalize_gray_8_or_16(img, bitdepth, wc=None, ww=None):
    if bitdepth not in (8, 16):
        raise AssertionError
    ww = np.max(img) - np.min(img) if ww is None else ww
    wc = (np.max(img) + np.min(img)) / 2 if wc is None else wc
    l = wc - ww / 2
    h = wc + ww / 2
    imga = img.copy()
    imga[img < l] = l
    imga[img > h] = h
    imga = np.interp(imga, (l, h), ((2 ** bitdepth) - 1, 0))
    return imga.astype(np.uint if bitdepth == 8 else np.uint16)
def normalize_gray_8_or_16_inverse(img, bitdepth, wc=None, ww=None):
    if bitdepth not in (8, 16):
        raise AssertionError
    ww = np.max(img) - np.min(img) if ww is None else ww
    wc = (np.max(img) + np.min(img)) / 2 if wc is None else wc
    l = wc - ww / 2
    h = wc + ww / 2
    imga = img.copy()
    imga[img < l] = l
    imga[img > h] = h
    imga = np.interp(imga, (l, h), (0, (2 ** bitdepth) - 1))
    return imga.astype(np.uint if bitdepth == 8 else np.uint16)


for root, dirs, files in walk(dir): 
    for f in files:
        a= a+1
        print(a)
        fullpath = join(root, f)
        print(fullpath)
        try:       
            dicom_info =dicom.read_file(fullpath)
            ds = dicom.dcmread(fullpath)

            img = ds.pixel_array
            img = np.array(img, dtype=np.uint16)

            ###check
            if(dicom_info.PhotometricInterpretation == 'MONOCHROME1'):
                img = normalize_gray_8_or_16(img, 16, dicom_info.WindowCenter, dicom_info.WindowWidth)
            else:
                img = normalize_gray_8_or_16_inverse(img, 16, dicom_info.WindowCenter, dicom_info.WindowWidth)

            if (hasattr(dicom_info,'PatientID')):
                pid = str(dicom_info.PatientID)
            else:
                pid=''
            if (hasattr(dicom_info,'PatientsAge')):
                age = int(dicom_info.PatientsAge[1:3])
            else:
                age=''
            if (hasattr(dicom_info,'StudyID')):
                sid = str(dicom_info.StudyID)
            else:
                sid=''
            if (hasattr(dicom_info,'StudyDate')):
                date = str(dicom_info.StudyDate)
            else:
                date=''
                
            print('AccessionNumber: ',dicom_info.AccessionNumber)
            print(dicom_info.ImageLaterality)
            print(dicom_info.ViewPosition)
            save_filename = raw_images_path+'AA0_'+str(dicom_info.AccessionNumber)+'_'+str(dicom_info.ImageLaterality)+'_'+str(dicom_info.ViewPosition)+'_1.png'
            if os.path.exists(save_filename):
                print('ERROR: already exists!')
            else:
                cv2.imwrite(save_filename, img)
      
            data={"patient_id":dicom_info.PatientID,"patient_age":age,"exam_date": dicom_info.StudyDate,"accession_number": dicom_info.AccessionNumber}
            with open(json_files_path+'BB0_'+dicom_info.AccessionNumber+'.json', 'w') as f:
                json.dump(data, f)

        except:
            print('File Name: ' + f + ', ERROR!')
            wrong_file.append(f)
            continue
print('wrong_file: ', wrong_file)