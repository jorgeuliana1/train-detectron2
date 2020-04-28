''' The source of this is script is https://gist.github.com/AlexeyGy/5e9c5a177db31569c20c76ad4dc39284
and it was authored by https://github.com/AlexeyGy, the access was made at April 27th, 2020.
'''
import json
import os
from cytoolz import merge, join, groupby
from cytoolz.compatibility import iteritems
from cytoolz.curried import update_in
from itertools import starmap
from collections import deque
from lxml import etree, objectify
from scipy.io import savemat
import imageio
from pathlib import Path
from tqdm import tqdm

imread = imageio.imread # The script originally used scipy.imread, but it is deprecated now.

def keyjoin(leftkey, leftseq, rightkey, rightseq):
    return starmap(merge, join(leftkey, leftseq, rightkey, rightseq))


def root(folder, filename, width, height):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
            E.folder(folder),
            E.filename(filename),
            E.source(
                E.database('MS COCO 2014'),
                E.annotation('MS COCO 2014'),
                E.image('Flickr'),
                ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(3),
                ),
            E.segmented(0)
            )


def instance_to_xml(anno):
    E = objectify.ElementMaker(annotate=False)
    xmin, ymin, width, height = anno['bbox']
    return E.object(
            E.name(anno['category_id']),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmin+width),
                E.ymax(ymin+height),
                ),
            )


def write_categories(coco_annotation, dst):
    content = coco_annotation
    categories = tuple( d['name'] for d in content['categories'] )
    savemat(os.path.abspath(dst), {'categories': categories})

def get_instances(coco_annotation):

    '''
        coco_annotation: Dictionary that represents the COCO annotation.
    '''
    
    categories = {d['id'] : d['name'] for d in coco_annotation['categories']}
    return categories, tuple(keyjoin('id', coco_annotation['images'], 'image_id', coco_annotation['annotations']))

        
def rename(name, year=2014):
        out_name = os.path.splitext(name)[0]
        return out_name


def create_imageset(annotations, dst):
    annotations = os.path.abspath(annotations)
    dst = os.path.abspath(dst)
    val_txt = os.path.join(dst,'val.txt')
    train_txt = os.path.join(dst, 'train.txt')

    for val in annotations.listdir('*val*'):
        val_txt.write_text('{}\n'.format(os.path.splitext(val.basename())[0]), append=True)

    for train in annotations.listdir('*train*'):
        train_txt.write_text('{}\n'.format(os.path.splitext(train.basename())[0]), append=True)

def create_annotations(coco_annotation, dst='annotations_voc'):


    os.makedirs(dst, exist_ok=True)

    categories, instances = get_instances(coco_annotation)

    '''
        About categories: Dictionary where the keys are the categories IDs and the values are tha categories names.
        About instances: Tuple of dictionaries containing information of the annotations and its respective images.
        NOTE: There is one instance for every annotation, not image.
    '''

    dst = os.path.abspath(dst)
   
    '''
        Modifying the category ID to show an string instead of a number.
        The string corresponds to the name of the category.
    '''
    for i, instance in tqdm(enumerate(instances),desc="rewriting categories"):
        instances[i]['category_id'] = categories[instance['category_id']]

    for name, group in tqdm(iteritems(groupby('file_name', instances)), total=len(groupby('file_name', instances)), desc="processing annotations"):
        
        '''
            About name: the image path
            About group: the image informations
        '''

        img = imread(os.path.abspath(name))
        if img.ndim == 3:
            out_name = rename(name)
            image_folder, image_name = os.path.split(out_name)
            annotation = root(image_folder, '{}.jpg'.format(image_name),  group[0]['height'], group[0]['width'])
            for instance in group:
                annotation.append(instance_to_xml(instance))

            # Exporting XML to destination folder
            destination_file = "{}.xml".format(out_name)
            _, destination_file = os.path.split(destination_file)
            xml_file = etree.ElementTree(annotation)
            xml_file.write(os.path.join(dst, destination_file))
            
