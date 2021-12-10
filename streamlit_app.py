import streamlit as st
import keras
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
import pandas as pd
import re
from PIL import Image


def load_model(modelfile):
    resnet50_url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'
    feature_extractor_layer = hub.KerasLayer(resnet50_url,
                                               trainable=False, # freeze the underlying patterns
                                               name='feature_extraction_layer',
                                               input_shape=(224, 224)+(3,)) # define the input image shap

    resnet_model2 = keras.Sequential([
        feature_extractor_layer,
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation = 'sigmoid'),
    ])

    metrics = ['accuracy', keras.metrics.AUC(), keras.metrics.Recall()]

    resnet_model2.compile(loss='binary_crossentropy',
                            metrics = [metrics],
                            optimizer = adam_v2.Adam(learning_rate = 5e-4))

    resnet_model2.load_weights(modelfile)
    return resnet_model2

def test_picture(model, path):
    df = pd.DataFrame(path, columns = ['Path'])
    df['Label'] = [re.findall('[0-9]{4}_(.+?).png', single_path)[0] for single_path in path]
    test_generator = ImageDataGenerator(rescale = 1./255,)
    test = test_generator.flow_from_dataframe(dataframe = df,
                                    class_mode  = 'binary',
                                    x_col       = 'Path',
                                    y_col       = 'Label',
                                    shuffle     = False,
                                    batch_size  = 32,
                                    target_size = (224, 224))
    temp = [(float(prob), True if prob > 0.33 else False) for prob in model.predict(test)]
    return [(path[i], temp[i][0], temp[i][1]) for i in range(len(path))]

st.set_page_config(page_title="Neural Network prediction on patient lungs", page_icon="https://www.freeiconspng.com/thumbs/heart-png/heart-png-15.png", layout='centered', initial_sidebar_state="collapsed")

def main(model):
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Neural Network prediction on patient lungs </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    paths_good = [
	        "png/MCUCXR_0001_0.png",
	        "png/MCUCXR_0002_0.png",
	        "png/MCUCXR_0003_0.png",
	        "png/MCUCXR_0004_0.png",
	        "png/MCUCXR_0005_0.png",
    ]
    paths_bad = [
	        "png/MCUCXR_0104_1.png",
	        "png/MCUCXR_0108_1.png",
	        "png/MCUCXR_0113_1.png",
	        "png/MCUCXR_0117_1.png",
	        "png/MCUCXR_0126_1.png",
    ]
    
    results = test_picture(model, paths_good + paths_bad)
    i = 0
    col_list_1 = st.columns(5)
    col_list_2 = st.columns(5)
    def prompt_result(result, st_place):
        image = Image.open(result[0])
        caption = "prob = " + str(result[1]) + "\nresult = " + str(result[2])
        st_place.image(image, caption=caption)
    for result in results:
        st_place = st
        if i < 10:
            st_place = col_list_2[i - 5]
        if i < 5:
            st_place = col_list_1[i]
        prompt_result(result, st_place)
        i += 1

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	model = load_model("model.h5")
	main(model)
