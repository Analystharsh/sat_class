import streamlit as st
import tensorflow as tf
import streamlit as st
import gdown
import imquality.brisque as brisque

@st.cache(allow_output_mutation=True)
def load_model():
  id = '1EELIjAOxt4gLpwa_PT27ptqZi9dFcFnf'
  output = 'my_model3.hdf5'
  gdown.download(id =id,output = output, quiet=False)
  model=tf.keras.models.load_model('my_model3.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Satellite Image Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
class_names = ['airplane', 'baseball_field', 'basketball_court', 'beach', 'bridge', 'cemetery', 'chaparral', 'christmas_tree_farm', 'closed_road', 'coastal_mansion', 'crosswalk', 'dense_residential', 'ferry_terminal', 'football_field', 'forest', 'freeway', 'golf_course', 'harbor', 'intersection', 'mobile_home_park', 'nursing_home', 'oil_gas_field', 'oil_well', 'overpass', 'parking_lot', 'parking_space', 'railway', 'river', 'runway', 'runway_marking', 'shipping_yard', 'solar_panel', 'sparse_residential', 'storage_tank', 'swimming_pool', 'tennis_court', 'transformer_station', 'wastewater_treatment_plant']
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (128,128)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    img = cv2.imread(file)
#     st.write(predictions)
#     st.write(score)
#     st.write(np.argmax(score))
    st.write(img.shape)
    st.write(brisque.score(img))
    st.write(
    "This image most likely belongs to {}."
    .format(class_names[np.argmax(score)])
)
#     st.write(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
    #print(
    #"This image most likely belongs to {} with a {:.2f} percent confidence."
    #.format(class_names[np.argmax(score)], 100 * np.max(score))
#)
