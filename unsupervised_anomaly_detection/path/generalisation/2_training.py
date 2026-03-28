"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from dotenv import load_dotenv
from utilities import data_class

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


for seed in range(1, 4):
    # Declare constants
    AD_MODE = "us"  # or 'ss'
    MODEL_NAME = (
        "tevae"  # or 'tcnae', 'omnianomaly', 'sisvae', 'lwvae', 'vsvae', 'vasp'
    )

    # Set fixed seed for random operations
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    # Load variables in .env file
    load_dotenv()

    # Load directory paths from .env file
    data_path = os.path.join(os.environ["data_path"], "path", "generalisation")
    model_path = os.path.join(os.environ["model_path"], "path", "generalisation")

    for fold_idx in range(7):
        # Declare model name and paths
        if AD_MODE == "us":
            data_load_path = os.path.join(data_path, "2_preprocessed", "unsupervised")
        else:
            data_load_path = os.path.join(data_path, "2_preprocessed", "semisupervised")
        model_save_path = os.path.join(
            model_path,
            MODEL_NAME + "_" + AD_MODE + "_" + str(fold_idx) + "_" + str(seed),
        )

        # Load data
        tfdata_train = tf.data.Dataset.load(
            os.path.join(data_load_path, "fold_" + str(fold_idx), "train")
        )
        tfdata_val = tf.data.Dataset.load(
            os.path.join(data_load_path, "fold_" + str(fold_idx), "val")
        )

        tfdata_train = tfdata_train.cache().batch(1024).prefetch(tf.data.AUTOTUNE)
        tfdata_val = tfdata_val.cache().batch(1024).prefetch(tf.data.AUTOTUNE)

        # Establish callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_rec_loss",
            mode="min",
            verbose=1,
            patience=250,
            restore_best_weights=True,
        )

        # Define model
        window_size = tfdata_train.element_spec.shape[1]  # type: ignore
        features = tfdata_train.element_spec.shape[2]  # type: ignore
        with strategy.scope():
            from model_garden.tevae import (
                KLAnnealing,
                TeVAE_Encoder,
                TeVAE_Decoder,
                MA,
                TeVAE,
            )

            annealing = KLAnnealing(
                annealing_type="cyclical",
                grace_period=25,
            )
            latent_dim = features // 2
            key_dim = features // 8
            hidden_units = features * 16
            encoder = TeVAE_Encoder(
                seq_len=window_size,
                latent_dim=latent_dim,
                features=features,
                hidden_units=hidden_units,
                seed=seed,
            )
            decoder = TeVAE_Decoder(
                seq_len=window_size,
                latent_dim=latent_dim,
                features=features,
                hidden_units=hidden_units,
                seed=seed,
            )
            ma = MA(
                seq_len=window_size,
                latent_dim=latent_dim,
                key_dim=key_dim,
                features=features,
            )
            model = TeVAE(encoder, decoder, ma)
            callback_list = [early_stopping, annealing]
            model.compile(
                optimizer=tf.keras.optimizers.Adam(amsgrad=True, clipnorm=None)
            )

        # Fit model
        history = model.fit(
            tfdata_train,
            epochs=10000,
            callbacks=callback_list,
            validation_data=tfdata_val,
            verbose=2,
        )

        # Run model on random data with same shape as input to build model
        model.predict(tf.random.normal((32, window_size, features)), verbose=0)

        # Save model and losses
        os.makedirs(model_save_path, exist_ok=True)
        model.save(os.path.join(model_save_path, "model.keras"))
        data_class.DataProcessor().dump_pickle(
            history, os.path.join(model_save_path, "losses.pkl")
        )
        tf.keras.backend.clear_session()
