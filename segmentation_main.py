from model_proposal import *
from data import *
import matplotlib.pyplot as plt

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, 'data/crack/train', 'image', 'label', data_gen_args, save_to_dir=None)
val_data = trainGenerator(2, 'data/crack/train', 'image', 'label', data_gen_args, save_to_dir=None)

model = model()
epochs = 400
model_checkpoint = ModelCheckpoint('test.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(myGene, validation_data=val_data, validation_steps=1, steps_per_epoch=1, epochs=epochs, callbacks=[model_checkpoint])

print(history.history.keys())
#  "Accuracy"
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Loss and Accuracy')
plt.ylabel('accuracy')
#plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
#plt.show()
#plt.savefig('Seg_acc.png')

# "Loss"
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
#plt.show()
plt.savefig('Segmentation_loss_acc.png')
'''
testGene = testGenerator("data/crack/test")
results = model.predict_generator(testGene, 354, verbose=1)

saveResult("data/crack/test", results)
'''