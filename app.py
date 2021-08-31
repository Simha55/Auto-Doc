import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import cv2
import numpy as np
smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)
@st.cache(allow_output_mutation=True)
def load_model(custom_objects):
  model1=tf.keras.models.load_model('resnet50_clf_model.hdf5')
  model2=tf.keras.models.load_model('brain_seg.hdf5',custom_objects)
  
  return model1, model2
with st.spinner('Model is being loaded..'):
  custom_objects = {"iou":iou, "dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef}
  model1, model2=load_model(custom_objects)

st.write("""
         # Auto-Doc
         """
         )
  

st.write("""
         # Brain Tumor Classification And Segmentation
         """
         )

file = st.file_uploader("Please upload an brain scan file")

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model1, model2):
    
        size = (256,256)    
        image1 = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image1 = np.asarray(image1)
        
        img1 = image1 / 255
        img1 = img1[np.newaxis, :, :, :]
        pred1 = model1.predict(img1)
        pred2 = model2.predict(img1)
        pred2 = np.squeeze(pred2) 
        return pred1, pred2
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    size = (256,256)    
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    st.image(image, use_column_width=False)
    predictions_1, predictions_2 = import_and_predict(image, model1, model2)
    st.image(predictions_2, use_column_width=False,clamp = True)
    class_names = ["No tumor", "Tumor"]
    
    string = "This image most likely have "+ class_names[np.argmax(predictions_1)]+" with a "+ str(100 * np.max(predictions_1)) +" percent confidence."
    st.success(string)

