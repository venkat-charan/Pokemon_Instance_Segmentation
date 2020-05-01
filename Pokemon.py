"""
Mask R-CNN


Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    #Train a new model starting from pre-trained COCO weights
    python pokemon.py train --dataset=/home/.../mask_rcnn/data/pokemon/ --weights=coco

    #Train a new model starting from pre-trained ImageNet weights
    python pokemon.py train --dataset=/home/.../mask_rcnn/data/pokemon/ --weights=imagenet

    # Continue training the last model you trained. This will find
    # the last trained weights in the model directory.
    python pokemon.py train --dataset=/home/.../mask_rcnn/data/pokemon/ --weights=last

    #Detect and color splash on a image with the last model you trained.
    #This will find the last trained weights in the model directory.
    python pokemon.py splash --weights=last --image=/home/...../*.jpg

    #Detect and color splash on a video with a specific pre-trained weights of yours.
    python sugery.py splash --weights=/home/.../logs/mask_rcnn_pokemon_0030.h5  --video=/home/simon/Videos/Center.wmv
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class pokemonConfig(Config):    
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pokemon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class pokemonDataset(utils.Dataset):
    def load_VIA(self, dataset_dir, subset, hc=False):
        """Load the Pokemon dataset from VIA.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or predict
        """
        # Add classes. We have only one class to add.
        self.add_class("pokemon", 1, "Pikachu")
        self.add_class("pokemon", 2, "Froakie")
        self.add_class("pokemon", 3, "Fletching")
        self.add_class("pokemon", 4, "Fennekin")
        self.add_class("pokemon", 5, "Dedenne")

        # Train or validation dataset?
        assert subset in ["train", "val"]
    
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {name:'a'},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

        annotations = list(annotations.values())  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            names = [r['region_attributes'] for r in a['regions'].values()]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "pokemon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a pokemon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "pokemon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        # In the pokemon dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        #print(class_names)
        for i, p in enumerate(class_names):
            #print(p)
        #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
            #print(p)
            if p['class_id'] == '1':
                class_ids[i] = 1
            elif p['class_id'] == '2':
                class_ids[i] = 2
            elif p['class_id'] == '3':
                class_ids[i] = 3
            elif p['class_id'] == '4':
                class_ids[i] = 4
            elif p['class_id'] == '5':
                class_ids[i] = 5
            #elif p['class_id'] == '6':
                #class_ids[i] = 6
            #elif p['class_id'] == '7':
             #   class_ids[i] = 7
            #elif p['class_id'] == '8':
            #    class_ids[i] = 8
            #elif p['class_id'] == '9':
            #    class_ids[i] = 9
            #elif p['class_id'] == '10':
            #    class_ids[i] = 10
            #elif p['class_id'] == '11':
            #    class_ids[i] = 11
            #elif p['class_id'] == '12':
            #    class_ids[i] = 12
            #elif p['class_id'] == '13':
             #   class_ids[i] = 13
            #elif p['class_id'] == '14':
                #class_ids[i] = 14

            #assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pokemon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model, *dic):
    """Train the model."""
    dataset_train = pokemonDataset()
    dataset_train.load_VIA(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = pokemonDataset()
    dataset_val.load_VIA(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedu le is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    import time

    start=time.time()
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0,0.5))
        ),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
        )
    ],random_order=True)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads',
                augmentation=augmentation)
    end=time.time()
    print("time taken is {}".format(end-start))


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None, out_dir=''):
    assert image_path or video_path

    #class_names = ['BG', 'Pikachu', 'Froakie','Fletching','Fennekin','Meowth','Dedenne','Squirtle','Conkeldurr','Psyduck','Black_bird','Bulbasaur','Chespin','Charamandar','Unknown']
    class_names = ['BG', 'Pikachu', 'Froakie','Fletching','Fennekin','Dedenne']

    # Image or video?
    if image_path:
        import cv2
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        
        image = skimage.io.imread(args.image)
        #print(image.shape)
        #cv2.imwrite('orig1.png',image)
        #rgbimage=skimage.color.rgba2rgb(image)
        rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        #cv2.imwrite('orig2.png',image)
        
        # Detect objects

        #cv2.imwrite("orig1.png",image.astype(np.uint8))
        r = model.detect([rgbimage], verbose=1)[0]
        # Color splash
        #splash = color_splash(image, r['masks'])
        mask=visualize.display_instances(rgbimage, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], making_image=True)
        #print(image.shape)
        #print(mask.shape)
        #ask=cv2.resize(mask,(512,512))
        print(mask.shape)
        #result=cv2.addWeighted(np.array(image),0.7,mask,0.3,0)
        #cv2.imwrite('result.png',result)
        file_name = 'splash.png'
        # Save output
        #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        #save_file_name = os.path.join(out_dir, file_name)
        #skimage.io.imsave(save_file_name, splash)

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        # width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = 1600
        height = 1600
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.wmv".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        #For video, we wish classes keep the same mask in frames, generate colors for masks
        colors = visualize.random_colors(len(class_names))
        while success:
            print("frame: ", count)
            # Read next image
            plt.clf()
            plt.close()
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                # splash = color_splash(image, r['masks'])

                splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                     class_names, r['scores'], colors=colors, making_video=True)
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect rings and robot arms.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/home/simon/mask_rcnn/data/pokemon",
                        help='Directory of the pokemon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/home/simon/logs/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = pokemonConfig()
    else:
        class InferenceConfig(pokemonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        print(weights_path)
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))



