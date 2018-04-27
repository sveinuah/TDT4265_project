#!python3

from matplotlib import pyplot as plt

def plot_training_score(history):
    print('Availible variables to plot: {}'.format(history.history.keys()))

    plt.figure(1)
    plt.plot([i for i in range(1,len(history.history['acc'])+1)],history.history['acc'])
    if 'val_acc' in history.history.keys():
        plt.plot([i for i in range(1,len(history.history['val_acc'])+1)],history.history['val_acc'])
        plt.legend(('Training data','Validation data'))

    else:
        plt.legend(('Training data'))

    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((0,1))


    plt.figure(2)
    plt.plot([i for i in range(1,len(history.history['acc'])+1)], history.history['loss'])

    if 'val_loss' in history.history.keys():
        plt.plot([i for i in range(1,len(history.history['val_loss'])+1)],history.history['val_loss'])
        plt.legend(('Training data','Validation data'))

    else:
        plt.legend(('Training data'))

    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(('Training data', 'validation data'))

    plt.show()


