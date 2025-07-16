import os
import numpy as np
import nibabel as nib
import ants
from sklearn.model_selection import train_test_split
from glob import glob
import tensorflow as tf
import random
from scipy.ndimage import rotate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout, Conv3DTranspose
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
# from seg_model.data import load_dataset, split_data, BraTSPEDGenerator
# from seg_model.model import unet_3d
# from seg_model.losses import dice_loss, iou_loss
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os   

def preprocess_brats_subject(subject_path):

    print(f"Preprocessing subject at {subject_path}")
    sequences = ['t2f', 't1n', 't1c', 't2w']
    data_3d = []

    for seq in sequences:
        img_path = glob(os.path.join(subject_path, f"*-{seq}.nii.gz"))[0]
        img = ants.image_read(img_path)
        img = ants.n4_bias_field_correction(img)
        img_np = img.numpy()

        # Remove low-intensity voxels & crop to center 144x144x144
        img_np = np.clip(img_np, 0, np.percentile(img_np, 99))  # remove dark voxels
        img_np = crop_center(img_np, (144, 144, 144))
        data_3d.append(img_np)

    # Stack all 4 sequences into one volume (4, 144, 144, 144)
    print(f"Shapes after preprocessing for {subject_path}: {[d.shape for d in data_3d]}")

    return np.stack(data_3d, axis=-1)# shape (4, 144, 144, 144)

def crop_center(img, target_shape):
    
    print("Inside crop_center function")
    center = [s // 2 for s in img.shape]
    crop_slices = tuple(slice(c - ts // 2, c + ts // 2) for c, ts in zip(center, target_shape))
    return img[crop_slices]

def load_dataset(root_folder):

    print(f"Loading dataset from {root_folder}")

    X, Y = [], []
    subjects = sorted(os.listdir(root_folder))
    print(f"Found {len(subjects)} subjects in {root_folder}")

    for subject in subjects[0:20]: # Limit to first 10 for debugging; change to subjects for full dataset
        print(f"Processing {subject}")

        sub_path = os.path.join(root_folder, subject)
        try:
            image = preprocess_brats_subject(sub_path)

            print(f"Image shape after preprocessing: {image.shape}")
            
            seg_path = glob(os.path.join(sub_path, "*-seg.nii.gz"))[0]
            seg = nib.load(seg_path).get_fdata()
            seg = crop_center(seg, (144, 144, 144))
            mask= to_categorical(seg, num_classes=5)
            X.append(image)
            Y.append(mask)  # one-hot encoded mask

            print(f"Segmentation shape after cropping:{mask.shape}")

        except Exception as e:
            print(f"Skipping {subject}: {e}")
            continue
    print(f"Loaded {len(X)} samples successfully.")
    print(f"Loaded {len(Y)} samples successfully.")
    print(f"Example image shape: {X[0].shape}, Example segmentation unique labels: {np.unique(Y[0])}")
    return np.array(X), np.array(Y)


def split_data(X, Y):

    print(f"Splitting data into train/val/test sets. Total samples: {len(X)}")

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=2/3, random_state=42)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


class BraTSPEDGenerator(Sequence):

    # Generates batches of 3D MRI data and corresponding segmentation masks for BraTS-PEDs dataset.
    # Applies random augmentations (flips, rotations) if augment=True.
    def __init__(self, X, Y, batch_size=2, augment=False):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.X))  # ← Required to avoid AttributeError

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        # Get indices for this batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = self.X[batch_indices]
        y_batch = self.Y[batch_indices]

        # Apply augmentation
        if self.augment:
            x_batch_aug = []
            y_batch_aug = []
            for x, y in zip(x_batch, y_batch):
                x, y = self.apply_augmentation(x, y)
                x_batch_aug.append(x)
                y_batch_aug.append(y)
            x_batch = np.array(x_batch_aug)
            y_batch = np.array(y_batch_aug)

        # Ensure correct dtype
        x_batch = x_batch.astype(np.float32)
        y_batch = y_batch.astype(np.float32)

        return x_batch, y_batch

kernel_initializer =  'he_uniform' #Try others if you want

################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model


def dice_loss(y_true, y_pred):
    print("Inside dice_loss function")
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def iou_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return 1 - ((intersection + smooth) / (union + smooth))

# for multiclass

def dice_score_per_class(y_true, y_pred, num_classes=4, smooth=1e-6):
    scores = []
    y_true = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), num_classes)
    y_pred = tf.one_hot(tf.cast(tf.squeeze(tf.round(y_pred)), tf.int32), num_classes)

    for c in range(1, num_classes):  # skip background (class 0)
        y_true_c = y_true[..., c]
        y_pred_c = y_pred[..., c]
        intersection = tf.reduce_sum(y_true_c * y_pred_c)
        union = tf.reduce_sum(y_true_c) + tf.reduce_sum(y_pred_c)
        dice = (2. * intersection + smooth) / (union + smooth)
        scores.append(dice)
    return tf.reduce_mean(scores)

def iou_score_per_class(y_true, y_pred, num_classes=4, smooth=1e-6):
    scores = []
    y_true = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), num_classes)
    y_pred = tf.one_hot(tf.cast(tf.squeeze(tf.round(y_pred)), tf.int32), num_classes)

    for c in range(1, num_classes):
        y_true_c = y_true[..., c]
        y_pred_c = y_pred[..., c]
        intersection = tf.reduce_sum(y_true_c * y_pred_c)
        union = tf.reduce_sum(y_true_c) + tf.reduce_sum(y_pred_c) - intersection
        iou = (intersection + smooth) / (union + smooth)
        scores.append(iou)
    return tf.reduce_mean(scores)


# 🔍 5. Grad-CAM (2D Slice Visualization)
def get_gradcam_heatmap(model, img, layer_name):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available for Grad-CAM")
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 🔬 6. Per-Layer Activation Visualization
def visualize_activations(model, input_volume, layers_to_show):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available for activation visualization")
    for layer_name in layers_to_show:
        sub_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        activations = sub_model.predict(input_volume[np.newaxis, ..., np.newaxis])
        num_filters = min(activations.shape[-1], 8)
        plt.figure(figsize=(15, 5))
        for i in range(num_filters):
            plt.subplot(1, num_filters, i+1)
            plt.imshow(activations[0, :, :, input_volume.shape[2]//2, i], cmap='viridis')
            plt.title(f"{layer_name}\nfilter {i}")
            plt.axis('off')
        plt.show()

# 🧪 7. SHAP for MRI Modality Attribution
def explain_shap(model, input_sample):
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not available")
    background = input_sample[:2]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(input_sample[2:3])
    shap_values = np.array(shap_values)[0, ..., 0]
    return shap_values

if __name__ == "__main__":
    
    print("\033[92mMain file preprocessing started...\033[0m")

    import os
    import numpy as np
    import nibabel as nib
    import ants
    from sklearn.model_selection import train_test_split
    from glob import glob
    import tensorflow as tf
    import random
    from scipy.ndimage import rotate
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout
    import tensorflow.keras.backend as K
    from tensorflow.keras.utils import Sequence
    # from seg_model.data import load_dataset, split_data, BraTSPEDGenerator
    # from seg_model.model import unet_3d
    # from seg_model.losses import dice_loss, iou_loss
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.callbacks import TensorBoard
    import datetime
    import os    
    
    print("TensorFlow version:", tf.__version__)

    print(tf.config.list_physical_devices('GPU'))  # Check if GPU is visible
    print(tf.config.list_physical_devices('CPU'))  # Check if CPU is visible
    if tf.config.list_physical_devices('GPU'):
        print("CUDNN detected!")
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(*load_dataset("data/raw/BraTS-PEDs2024_Training"))
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train: {Y_train.shape}, Val: {Y_val.shape}, Test: {Y_test.shape}")
    print(f"Train labels unique: {np.unique(Y_train)}, Val labels unique: {np.unique(Y_val)}, Test labels unique: {np.unique(Y_test)}") 

    train_gen = BraTSPEDGenerator(X_train, Y_train)
    val_gen = BraTSPEDGenerator(X_val, Y_val)

    # Print shapes of first batch from generators
    x_batch, y_batch = train_gen[0]
    print(f"Train generator batch x shape: {x_batch.shape}, y shape: {y_batch.shape}")
    x_val_batch, y_val_batch = val_gen[0]
    print(f"Val generator batch x shape: {x_val_batch.shape}, y shape: {y_val_batch.shape}")

    ###########################################################################
    #Define loss, metrics and optimizer to be used for training
    # wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
    # # Convert class weights to tensor before passing
    # class_weights = tf.constant([0.20,0.20,0.20,0.20,0.20], dtype=tf.float64)  # for 5 classes (0-4)

    # import segmentation_models_3D as sm
    # dice_loss = sm.losses.DiceLoss(class_weights=class_weights)  # Dice loss with class weights
    # focal_loss = sm.losses.CategoricalFocalLoss()
    # total_loss = dice_loss + (1 * focal_loss)

    # metrics = ['accuracy']

    LR = 0.0001
    # optim = keras.optimizers.Adam(LR)
    #######################################################################
    #Fit the model 

    # steps_per_epoch = len(train_img_list)//batch_size
    # val_steps_per_epoch = len(val_img_list)//batch_size


    # from  simple_3d_unet import simple_unet_model

    model = simple_unet_model(IMG_HEIGHT=144, 
                            IMG_WIDTH=144, 
                            IMG_DEPTH=144, 
                            IMG_CHANNELS=4, 
                            num_classes=5)

    model.compile(optimizer = 'adam', loss=dice_loss,metrics=['accuracy'])  # Add dice_score_per_class, iou_score_per_class if needed

    print(model.input_shape)
    print(model.output_shape)

    history=model.fit(train_gen, validation_data=val_gen, epochs=50)

    model.save('brats_3d.hdf5')

    test_gen = BraTSPEDGenerator(X_test, Y_test)
    results = model.evaluate(test_gen)
    print(f"Test Loss: {results[0]} | Test Accuracy: {results[1]}")

    for i in range(3):
        x = X_test[i][np.newaxis, ...]  # Add batch dimension
        y_true = Y_test[i]
        y_pred = model.predict(x)[0, ..., 0]

        print(f"Sample {i+1}: Dice={1 - dice_loss(y_true, y_pred):.4f}")
