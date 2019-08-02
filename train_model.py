import configuration as config
import Models, LoadBatches
from keras.optimizers import RMSprop
import glob


def train_model():

    train_images_path = config.TRAIN_IMAGES_PATH
    train_segs_path = config.TRAIN_SEGMENTATION_PATH
    train_batch_size = config.TRAIN_BATCH_SIZE
    n_classes = config.NUM_CLASSES
    input_height = config.INPUT_HEIGHT
    input_width = config.INPUT_WIDTH
    validate = config.VALIDATE
    save_weights_path = config.SAVE_WEIGHTS_PATH
    epochs = config.EPOCHS
    load_weights = config.LOAD_WEIGHTS
    model_name = config.MODEL_NAME


    if validate:
        val_images_path = config.VAL_IMAGES_PATH
        val_segs_path = config.VAL_SEGS_PATH
        val_batch_size = config.VAL_BATCH_SIZE

    trainNumImages = glob.glob(train_images_path + "*.jpg") + glob.glob(train_images_path + "*.png") + glob.glob(train_images_path + "*.jpeg")
    trainNumImages.sort()

    valNumImages = glob.glob(val_images_path + "*.jpg") + glob.glob(val_images_path + "*.png") + glob.glob(val_images_path + "*.jpeg")
    valNumImages.sort()

    modelFns = {'vgg_unet': Models.VGGUnet.VGGUnet}
    modelFN = modelFns[model_name]

    m = modelFN(n_classes, input_height=input_height, input_width=input_width)

    opt = RMSprop(lr=0.0001)
    # opt = SGD(lr=0.001)

    m.compile(loss='binary_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])


    if len( load_weights ) > 0:
        m.load_weights(load_weights)

    print("Model output shape"), m.output_shape

    output_height = m.outputHeight
    output_width = m.outputWidth

    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path,  train_batch_size,  n_classes,
                                                input_height, input_width, output_height, output_width)

    if validate:
        G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes,
                                                      input_height, input_width, output_height, output_width)

    # === save weights and models after earch epoch
    for ep in range(epochs):
        m.fit_generator(G, steps_per_epoch=len(trainNumImages) // config.TRAIN_BATCH_SIZE,
                        validation_data=G2, validation_steps=len(valNumImages) // config.VAL_BATCH_SIZE,
                        epochs=1, verbose=1)
        m.save_weights(save_weights_path + "ex." + str(ep))
        m.save(save_weights_path + "m.model." + str(ep))
