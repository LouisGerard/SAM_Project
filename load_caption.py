import numpy as np
import json
import keras
from nltk.tokenize import word_tokenize, sent_tokenize
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")

def extract_features_caption(img_folder="data/train2014/",
                            caption_file='data/annotations/captions_train2014.json',
                            save_captions=True,
                            _type="train",
                            save_features=True,
                            save_all=False):
    features = []
    captions = []
    i = 0
    if save_captions:
        open("./"+_type+".captions.txt", 'w').close()
    with open('data/annotations/captions_train2014.json', 'r') as f:
        data = json.load(f)
    for img in data['images']:
        im = image.load_img(img_folder+img['file_name'], target_size=(224, 224))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        found = False
        for annotation in data['annotations']:
            if not found:
                if annotation['image_id'] == img['id']:
                    words = word_tokenize(annotation['caption'])
                    words=[word.lower() for word in words if word.isalpha()]
                    features.append(np.expand_dims(resnet_model.predict(x).flatten(), axis=0))
                    captions.append(' '.join(words))
                    if save_captions:
                        with open("./"+_type+".captions.txt",'a') as fp:
                            fp.write(str(img['file_name'])+'\t'+(' '.join(words))+'\n')
                    if not save_all:
                        found = True
                        break
                    i+=1
                    
                    if i % 500 == 0:
                        print("processed {}".format(i))
        
    features = np.array(features)
    features = features.reshape((-1,2048))
    np.save('resnet50-features.train.40k.npy',features)
    return features,captions



    