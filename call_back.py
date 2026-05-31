import tensorflow as tf 

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        
        if logs["accuracy"] >= 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True
