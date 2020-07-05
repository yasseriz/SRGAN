def build_autoencoder():

    input_shape = (256, 256, 3)
    input_layer = Input(shape=input_shape)
    # Auto-Encoder
    l1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_layer)
    l2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
    l3 = MaxPooling2D(padding='same')(l2)

    l4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
    l6 = MaxPooling2D(padding='same')(l5)

    l7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)
    # at this point the representation is (7, 7, 32)

    l8 = UpSampling2D()(l7)
    l9 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                activity_regularizer=tf.keras.regularizers.l1(10e-10))(l8)
    l10 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                 activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)

    l11 = Add(name='encode')([l10, l5])

    l12 = UpSampling2D((2, 2))(l11)

    l13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                 activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
    l14 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu',
                 activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)
    l15 = Add(name='decode')([l14, l2])
    decoded = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='tanh',
                     activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[decoded],
                  name='autoencoder')

    return model