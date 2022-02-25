import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from autoencoders.tools import repeat_block
import pathlib
from joblib import dump, load


def save_model(model, input_shape, latent_dim, autoencoder_kwargs, path="weights", verbose=False):
    path = pathlib.Path(path)
    model_kwargs = {"input_shape": input_shape, "latent_dim": latent_dim, "autoencoder_kwargs": autoencoder_kwargs}
    model.save_weights(path.parents[0] / (path.name + ".h5"))
    dump(model_kwargs, path.parents[0] / (path.name + ".joblib"))
    return


def load_model(path="weights", verbose=False):
    path = pathlib.Path(path)
    model_kwargs = load(path.parents[0] / (path.name + ".joblib"))
    encoder, decoder, autoencoder = build_vae(
        model_kwargs.get("input_shape"),
        model_kwargs.get("latent_dim"),
        model_kwargs.get("autoencoder_kwargs"),
        verbose=verbose
    )
    autoencoder.load_weights(path.parents[0] / (path.name + ".h5"))
    return encoder, decoder, autoencoder


def build_encoder(
        input_shape,
        latent_dim,
        **kwargs
):
    encoder_input = keras.layers.Input(shape=(input_shape))
    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(latent_dim), scale=1.),
                                          reinterpreted_batch_ndims=1)
    encoder_layers = repeat_block(
        encoder_input,
        "encoder",
        filters=kwargs.get("filters"),
        kernel_size=kwargs.get("kernel_size"),
        padding=kwargs.get("padding"),
        activation=kwargs.get("activation"),
        n_layers=kwargs.get("n_layers"),
        pooling=kwargs.get("pooling"),
        batch_normalization=kwargs.get("batch_normalization"),
        n_layers_residual=kwargs.get("n_layers_residual")
    )
    encoder_layers = keras.layers.Conv1D(
        filters=input_shape[1],
        kernel_size=1,
        padding="same")(encoder_layers)
    encoder_layers = keras.layers.Flatten()(encoder_layers)
    encoder_layers = keras.layers.Dense(tfp.layers.IndependentNormal.params_size(latent_dim), activation=None,
                                        name='z_params')(encoder_layers)
    encoder_output = tfp.layers.IndependentNormal(
        latent_dim, convert_to_tensor_fn=tfp.distributions.Distribution.sample,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=kwargs.get("kl_weight", 1)),
        name='z_layer')(encoder_layers)
    encoder = keras.Model(encoder_input, encoder_output, name="Encoder")
    return encoder


def build_decoder(
        encoder,
        latent_dim,
        **kwargs
):
    decoder_input = keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    decoder_layers = decoder_input
    decoder_layers = keras.layers.Dense(encoder.layers[-3].output_shape[1])(decoder_layers)
    decoder_layers = keras.layers.Reshape(encoder.layers[-4].output_shape[1:])(decoder_layers)

    decoder_layers = repeat_block(
        decoder_layers,
        "decoder",
        filters=kwargs.get("filters")[::-1],
        kernel_size=kwargs.get("kernel_size")[::-1],
        padding=kwargs.get("padding")[::-1],
        activation=kwargs.get("activation")[::-1],
        n_layers=kwargs.get("n_layers"),
        pooling=kwargs.get("pooling")[::-1],
        batch_normalization=kwargs.get("batch_normalization"),
        n_layers_residual=kwargs.get("n_layers_residual")
    )
    decoder_output = keras.layers.Conv1D(
        filters=encoder.input_shape[2],
        kernel_size=1,
        padding="same")(decoder_layers)
    decoder = keras.models.Model(decoder_input, decoder_output, name="Decoder")
    return decoder



def build_vae(
        input_shape,
        latent_dim,
        autoencoder_kwargs,
        verbose=True,
):
    encoder = build_encoder(
        input_shape,
        latent_dim,
        **autoencoder_kwargs
    )

    decoder = build_decoder(
        encoder,
        latent_dim,
        **autoencoder_kwargs
    )

    model_input = keras.layers.Input(shape=input_shape)
    model_output = decoder(encoder(model_input))

    autoencoder = keras.models.Model(model_input, model_output, name="VAE")
    autoencoder.compile(
        optimizer=autoencoder_kwargs.get("optimizer", keras.optimizers.Adam()),
        loss=autoencoder_kwargs.get("loss", "mse"), metrics=["mse"])

    if verbose:
        encoder.summary()
        decoder.summary()
        autoencoder.summary()

    return encoder, decoder, autoencoder


if __name__ == "__main__":
    from autoencoders.plots import plot_grouped_history, plot_latent_space, plot_reconstruction_vae, \
        plot_labeled_latent_space_matrix
    from autoencoders.cached_datasets import load_cbf

    X_train, y_train, X_test, y_test = load_cbf()

    input_shape = X_train.shape[1:]

    latent_dim = 2

    autoencoder_kwargs = {
        "filters": [4, 4, 4, 4],
        "kernel_size": [3, 3, 3, 3],
        "padding": ["same", "same", "same", "same"],
        "activation": ["relu", "relu", "relu", "relu"],
        "pooling": [1, 1, 1, 1],
        "n_layers": 4,
        "optimizer": keras.optimizers.Adam(lr=0.001),
        "n_layers_residual": None,
        "batch_normalization": None,
        "kl_weight": 0.1

    }

    encoder, decoder, autoencoder = build_vae(input_shape, latent_dim, autoencoder_kwargs)
    hist = autoencoder.fit(X_train, X_train, epochs=2000, validation_split=0.2)
    plot_grouped_history(hist.history)
    plot_reconstruction_vae(X_train[:5], encoder, decoder)
    plot_latent_space(encoder.predict(X_train), y_train)
    plot_labeled_latent_space_matrix(encoder.predict(X_train), y_train)
