from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os # Import os module

from DataGenerator import ADSADataGenerator # your custom generator

def create_model(input_image_shape=(512, 640, 3), input_param_size=2, freeze_until=100):
    """
    MobileNetV2 for regression with numeric inputs.
    """
    img_input = Input(shape=input_image_shape, name="img_input")
    param_input = Input(shape=(input_param_size,), name="param_input")

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(input_shape=input_image_shape, include_top=False, weights='imagenet')

    # Freeze first N layers
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until

    x = base_model(img_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Custom trainable layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)

    # Concatenate with numeric input
    combined = Concatenate()([x, param_input])
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.2)(z)
    output = Dense(1, activation='linear')(z)

    model = Model(inputs=[img_input, param_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    return model

def main():
    #dataset_path = "/content/drive/MyDrive/DataSetCombined"
    dataset_path = "/home/jordanw7/koa_scratch/ADSA-AI/DataSetCombined"
    batch_size = 16
    image_size = (512, 640)

    # Print paths for debugging
    print(f"Image directory path: {os.path.join(dataset_path, 'Edges')}")
    print(f"Output CSV path: {os.path.join(dataset_path, 'output_params.csv')}")


    train_gen = ADSADataGenerator(dataset_path, split='train', batch_size=batch_size,
                              image_size=image_size, output_type='Surface Tension (mN/m)')
    val_gen = ADSADataGenerator(dataset_path, split='val', batch_size=batch_size,
                                image_size=image_size, output_type='Surface Tension (mN/m)')
    test_gen = ADSADataGenerator(dataset_path, split='test', batch_size=batch_size,
                                image_size=image_size, output_type='Surface Tension (mN/m)')

    # Model now expects 3 channels
    model = create_model(input_image_shape=(512, 640, 3), input_param_size=2, freeze_until=100)

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=50,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    # Save model
    model.save("SurfaceTension_Model.h5")

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Plot training/validation curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title("Mean Absolute Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()