import tensorflow as tf 

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        
        if logs["accuracy"] >= 0.99:
            print("\nModel reached 99% accuracy stopping the training")
            self.model.stop_training = True
