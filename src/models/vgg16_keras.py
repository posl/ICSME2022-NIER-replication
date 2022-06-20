from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

model = VGG16( weights='imagenet', include_top=True )
img_path = '1004.jpg'
img = image.load_img( img_path, target_size=(224, 224) )
x = image.img_to_array( img )
x = np.expand_dims(x, axis=0)

model.summary()

preds = model.predict( preprocess_input(x) )
print( preds )
results = decode_predictions( preds, top=5)[0]
for result in results: 
    print(result)
    
