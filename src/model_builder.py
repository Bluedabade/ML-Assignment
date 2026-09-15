"""MobileNetV2 พร้อม classification head ที่ปรับจาก config."""
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from src.config import *
def compile_model(model,lr): model.compile(optimizer=RMSprop(learning_rate=lr),loss="categorical_crossentropy",metrics=["accuracy"])
def build_model(weights="imagenet"):
    # include_top=False คือการตัด Head เดิมออก
    base=MobileNetV2(input_shape=(*IMAGE_SIZE,3),include_top=False,weights=weights)
    base.trainable=False # Freeze weight ของ pretrained model
    layers=[base,GlobalAveragePooling2D(name="global_average_pooling")]
    for i,units in enumerate(HEAD_DENSE_UNITS):
        layers.append(Dense(units,activation="relu",name=f"head_dense_{i+1}"))
        rate=(DROPOUT_RATES[0] if len(DROPOUT_RATES)==1 else DROPOUT_RATES[i]) if DROPOUT_RATES else 0
        if rate: layers.append(Dropout(rate,name=f"head_dropout_{i+1}"))
    layers.append(Dense(NUM_CLASSES,activation="softmax",name="skin_probabilities"))
    model=Sequential(layers,name="mobilenetv2_skin_5class"); compile_model(model,LEARNING_RATE); return model,base
def enable_fine_tuning(model,base,lr=FINE_TUNE_LEARNING_RATE):
    base.trainable=True
    for layer in base.layers[:FINE_TUNE_AT]: layer.trainable=False
    compile_model(model,lr)
