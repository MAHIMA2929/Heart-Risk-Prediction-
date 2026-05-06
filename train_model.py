from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

INIT_LR = 1e-4
EPOCHS = 5
BS = 32

# Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=BS,
    class_mode="binary",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=BS,
    class_mode="binary",
    subset="validation"
)

print("CLASS INDEX:", train_data.class_indices)

# Load Base Model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=(224, 224, 3))

# Custom Head
x = baseModel.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=baseModel.input, outputs=x)

# Freeze Base Layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=INIT_LR),
    metrics=["accuracy"]
)

# Train
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save Model (.keras format)
os.makedirs("model", exist_ok=True)
model.save("model/mask_model.keras")

print("✅ Model Saved Successfully!")