import os
import sys
import cv2
import argparse
import ast
from insightface_func.face_detect_crop_single import Face_detect_crop
from pathlib import Path
from tqdm import tqdm

def main(args):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 1, det_thresh=0.6, det_size=(640,640))
    crop_size = 224

    dirs= os.listdir(args.path_to_dataset)
    for i, dir in enumerate(tqdm(dirs)): #this is for the main dir
        
        #set the saving dir for all same identity
        identity_id=dir[:7]# just get the identity name
        dir_to_save= os.path.join(args.save_path,identity_id)
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
        
        sub_dir=os.path.join(args.path_to_dataset,dir)
        clips=os.listdir(sub_dir)
        for clip in clips: # this is inside identity/ clips/ 
            
            clip_dir=os.path.join(sub_dir,clip)
            image_names = os.listdir(clip_dir)
            for image_name in image_names:
                
                try:
                    image_path = os.path.join(clip_dir,image_name)
                    image = cv2.imread(image_path)
                    if args.pad_value:
                        image=app.padding(image,ast.literal_eval(args.pad_value))
                    cropped_image, _ = app.get(image, crop_size)
                    cv2.imwrite(os.path.join(dir_to_save, image_name), cropped_image[0])
                except:
                    pass
                
        
        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='./VggFace2/VGG-Face2/data/preprocess_train', type=str)
    parser.add_argument('--save_path', default='./VggFace2-crop', type=str)
    
    parser.add_argument('--pad_value', default=None)
    
    args = parser.parse_args()
    
    main(args)
