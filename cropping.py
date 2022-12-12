import os
import json
import xmltodict
from PIL import Image, ImageDraw
import sys
import pandas as pd
import multiprocessing
import glob
from tqdm import tqdm
from functools import partial
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Cropping images')
parser.add_argument('--imgs_path', default='imagenet/val/', type=str, help='path to images to be cropped')
parser.add_argument('--xml_dir', default='ILSVRC2012_bbox_val_v3/val/', type=str, help='path to bounding boxes in xml files')
parser.add_argument('--output_folder', default='imagenet1k_val_cropped/', type=str, help='path to output cropped images')
parser.add_argument('--flash_classes', default=False, type=bool, help='Whether to crop all ImageNet validation images or just the 50 classes from the image-flashing experiments')
args = parser.parse_args()

img_folder= args.imgs_path
xml_dir = args.xml_dir
output_folder = args.output_folder
flash_classes = args.flash_classes

multiple_bbox_count = 0

if not flash_classes:
    img_directories = os.listdir(img_folder)
else:
    img_directories = ['n04133789', 'n03995372', 'n03874599', 'n07753592', 'n07749582', 'n04039381', 'n03937543', 'n03197337', 'n03041632', 'n03814906', 'n03637318', 'n03729826', 'n04399382', 'n03938244', 'n04153751', 'n03063599', 'n04131690', 'n03595614', 'n03223299', 'n03255030', 'n03793489', 'n03958227', 'n04548362', 'n04208210', 'n04507155', 'n04270147', 'n04127249', 'n03891251', 'n03481172', 'n03291819', 'n04404412', 'n04118776', 'n03584829', 'n04004767', 'n04579432', 'n02786058', 'n04356056', 'n03887697', 'n03876231', 'n02992529', 'n04265275', 'n04522168', 'n03483316', 'n04332243', 'n04557648', 'n02747177', 'n03970156', 'n02769748', 'n03691459', 'n03676483']
    
img_directory_to_labels = {}
for directory in tqdm(img_directories):
    labels = {}
    img_names = set(os.listdir(os.path.join(img_folder, directory)))
    
    for img_name in img_names:
        img_xml_fname = img_name.split('.')[0] + '.xml'
        xml_file_path = xml_dir + img_xml_fname

        if not os.path.isfile(xml_file_path):
            continue
            
        with open(xml_file_path) as xml:
            xml_dict = xmltodict.parse(xml.read())
        xml.close()

        if type(xml_dict['annotation']['object']) != list:
            xmin = int(xml_dict['annotation']['object']['bndbox']['xmin'])
            ymin = int(xml_dict['annotation']['object']['bndbox']['ymin'])
            xmax = int(xml_dict['annotation']['object']['bndbox']['xmax'])
            ymax = int(xml_dict['annotation']['object']['bndbox']['ymax'])
        else:
            '''
            if more than one bbox's, select the one that's closest to the center of the img
            '''
            multiple_bbox_count += 1
            xmin, ymin, xmax, ymax = None, None, None, None
            im = Image.open(os.path.join(img_folder, directory, img_name))
            width, height = im.size
            x_center, y_center = width/2, height/2
            dist = float('inf')
            for box in xml_dict['annotation']['object']:
                bbox_xmin = int(box['bndbox']['xmin'])
                bbox_ymin = int(box['bndbox']['ymin'])
                bbox_xmax = int(box['bndbox']['xmax'])
                bbox_ymax = int(box['bndbox']['ymax'])
                bbox_x_center = bbox_xmin+(bbox_xmax-bbox_xmin)/2
                bbox_y_center = bbox_ymin+(bbox_ymax-bbox_ymin)/2
                if (bbox_x_center-x_center)**2 + (bbox_y_center-y_center)**2 < dist:
                    xmin, ymin, xmax, ymax = bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax
        labels[img_name] = {
                            'xmin':xmin, 
                            'ymin':ymin, 
                            'xmax':xmax, 
                            'ymax':ymax
                            }
    img_directory_to_labels[directory] = labels



print("# images w/ more than 1 bbox: " + str(multiple_bbox_count))



def cropping(img_directory, img_name, target=None, final_side_length=None, img_dir_to_labels=None):
    labels = img_dir_to_labels[img_directory]
    annotated_top = labels[img_name]['ymin']
    annotated_left = labels[img_name]['xmin']
    annotated_bottom = labels[img_name]['ymax']
    annotated_right = labels[img_name]['xmax']
    annotated_height = annotated_bottom - annotated_top
    annotated_width = annotated_right - annotated_left

    im = Image.open(os.path.join(img_folder, img_directory, img_name))
    img_width, img_height = im.size



    # crop each image such that the center of the bounding box is the center of the image
    crop_top, crop_left, crop_height, crop_width = 0, 0, 0, 0
    if annotated_top > img_height - annotated_bottom:
        crop_top = annotated_top - (img_height - annotated_bottom)
        crop_height = annotated_height + 2*(img_height - annotated_bottom)
    else:
        crop_top = 0
        crop_height = annotated_height + 2*annotated_top

    if annotated_left > img_width - annotated_right:
        crop_left = annotated_left - (img_width - annotated_right)
        crop_width = annotated_width + 2*(img_width - annotated_right)
    else:
        crop_left = 0
        crop_width = annotated_width + 2*annotated_left

    crop_right = crop_left + crop_width
    crop_bottom = crop_top + crop_height



    # crop to make image as square as possible
    lower_percent, upper_percent = None, None
    cropped_side_length, cropped_bbox_height, cropped_bbox_width = None, None, None
    run_into_edge = None
    
    sq_crop_left, sq_crop_right, sq_crop_top, sq_crop_bottom = crop_left, crop_right, crop_top, crop_bottom
    tar_crop_left, tar_crop_right, tar_crop_top, tar_crop_bottom = None, None, None, None
    
    if crop_height > crop_width:
        original_border_height = (crop_height - annotated_height)/2
        if annotated_height > crop_width:
            center = (crop_top + crop_bottom)/2
            sq_crop_top = center - crop_width/2
            sq_crop_bottom = center + crop_width/2

            # get lower and upper percent bound
            lower_percent = annotated_width/crop_width
            upper_percent = 1
            
            # crop to target percent ---- x = side length to reduce
            # annotated_width*(crop_width-2x)/(crop_width-2x)**2 = tar
            side_to_reduce = (annotated_width/target-crop_width)/-2
            tar_crop_left = sq_crop_left + side_to_reduce
            tar_crop_right = sq_crop_right - side_to_reduce
            tar_crop_top = sq_crop_top + side_to_reduce
            tar_crop_bottom = sq_crop_bottom - side_to_reduce
            
            cropped_side_length = tar_crop_bottom - tar_crop_top
            cropped_bbox_height = cropped_side_length
            cropped_bbox_width = annotated_width
            run_into_edge = True
            
        else:
            new_border_height = (crop_width - annotated_height)/2
            sq_crop_top =  crop_top + original_border_height - new_border_height
            sq_crop_bottom = crop_bottom + new_border_height - original_border_height

            # get lower and upper percent bound
            sq_crop_height = sq_crop_bottom - sq_crop_top
            lower_percent = sq_crop_height*annotated_width/(crop_width**2)
            to_reduce = min(sq_crop_height-annotated_height, crop_width-annotated_width)
            upper_percent = (sq_crop_height - to_reduce) * (crop_width - to_reduce)/(crop_width**2)

            # crop to target percent ---- x = side length to reduce
            # (annotated_height * annotated_width)/(sq_crop_height-2x)**2 = tar
            side_to_reduce = ((annotated_width*annotated_height/target)**0.5 - sq_crop_height)/(-2)
            tar_crop_left = sq_crop_left + side_to_reduce
            tar_crop_right = sq_crop_right - side_to_reduce
            tar_crop_top = sq_crop_top + side_to_reduce
            tar_crop_bottom = sq_crop_bottom - side_to_reduce
            
            cropped_side_length = tar_crop_bottom - tar_crop_top
            cropped_bbox_height = annotated_height
            cropped_bbox_width = annotated_width
            run_into_edge = False
            
            
    else:
        original_border_width = (crop_width - annotated_width)/2
        if annotated_width > crop_height:
            center = (crop_left + crop_right)/2
            sq_crop_left = center - crop_height/2 
            sq_crop_right = center + crop_height/2

            # cropping stuff from the bbox to make it a square
            lower_percent = annotated_height/crop_height
            upper_percent = 1
            
            # crop to target percent ---- x = side length to reduce
            # annotated_height*(crop_height-2x)/(crop_height-2x)**2 = tar
            side_to_reduce = (annotated_height/target-crop_height)/-2
            tar_crop_left = sq_crop_left + side_to_reduce
            tar_crop_right = sq_crop_right - side_to_reduce
            tar_crop_top = sq_crop_top + side_to_reduce
            tar_crop_bottom = sq_crop_bottom - side_to_reduce
            
            cropped_side_length = tar_crop_bottom - tar_crop_top
            cropped_bbox_height = annotated_height
            cropped_bbox_width = cropped_side_length
            run_into_edge = True
            
        else:
            new_border_width = (crop_height - annotated_width)/2
            sq_crop_left = crop_left + original_border_width - new_border_width
            sq_crop_right = crop_right + new_border_width - original_border_width

            # get lower and upper percent bound
            sq_crop_width = sq_crop_right - sq_crop_left
            lower_percent = sq_crop_width*annotated_height/(crop_height**2)
            to_reduce = min(sq_crop_width-annotated_width, crop_height-annotated_height)
            upper_percent = (sq_crop_width - to_reduce) * (crop_height - to_reduce)/(crop_height**2)

            # crop to target percent ---- x = side length to reduce
            # (annotated_height * annotated_width)/(sq_crop_width-2x)**2 = tar
            side_to_reduce = ((annotated_width*annotated_height/target)**0.5 - sq_crop_width)/(-2)
            tar_crop_left = sq_crop_left + side_to_reduce
            tar_crop_right = sq_crop_right - side_to_reduce
            tar_crop_top = sq_crop_top + side_to_reduce
            tar_crop_bottom = sq_crop_bottom - side_to_reduce
            
            cropped_side_length = tar_crop_bottom - tar_crop_top
            cropped_bbox_height = annotated_height
            cropped_bbox_width = annotated_width
            run_into_edge = False
    

    # scale all images such that each image has the same size    
    im1 = im.crop((round(tar_crop_left), round(tar_crop_top), round(tar_crop_right), round(tar_crop_bottom)))
    scaling_factor = final_side_length/cropped_side_length
    scaled_width = round(cropped_side_length*scaling_factor)
    scaled_height = round(cropped_side_length*scaling_factor)
    im1 = im1.resize((scaled_width, scaled_height))
    
    if not os.path.exists(os.path.join(output_folder, img_directory)):
        os.makedirs(os.path.join(output_folder, img_directory))
    im1.save(os.path.join(output_folder, img_directory, img_name))
    
    df = pd.DataFrame(columns=['image_name', 
                               'class_name', 
                               'lower_percent', 
                               'upper_percent', 
                               'original_img_height', 
                               'original_img_width', 
                               'cropped_img_side_length',
                               'original_bbox_height',
                               'original_bbox_width',
                               'cropped_bbox_height',
                               'cropped_bbox_width',
                               'run_into_edge'])
    
    df.loc[0] = [img_name, 
                 img_directory, 
                 lower_percent, 
                     upper_percent, 
                     img_height, 
                     img_width, 
                     cropped_side_length, 
                     annotated_height, 
                     annotated_width, 
                     cropped_bbox_height, 
                     cropped_bbox_width, 
                     run_into_edge]
    return df


# multiprocess cropping
img_names, img_classes = [], []
for cls in os.listdir(img_folder):
    if cls not in img_directories:
        continue
    for img in os.listdir(os.path.join(img_folder, cls)):
        if '.ipynb' in img:
            continue
        img_classes.append(cls)
        img_names.append(img)

args = [(img_classes[i], img_names[i]) for i in range(len(img_names))]

pool = multiprocessing.Pool(multiprocessing.cpu_count())
target_percent = 0.9202898550724637
final_img_side_length = 224
func = partial(cropping, target=target_percent, final_side_length=final_img_side_length, img_dir_to_labels=img_directory_to_labels)

results = []
for result in tqdm(pool.starmap(func, args), total=len(img_names)):
    results.append(result)
pool.close()
