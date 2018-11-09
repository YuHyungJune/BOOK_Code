import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# gan_mnist의 EPOCHS값
EPOCHS = 100

def display(fname):
    img = Image.open(fname)
    img.show()

def result(ep):
    with open('save_gimage.pkl', 'rb') as f:
        save_gimage = pkl.load(f)
        
        fig, axes = plt.subplots(5, 5, figsize=(28,28))
        
        for img, ax in zip(save_gimage[ep], axes.flatten()):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img.reshape((28,28)), cmap='gray')
        if ep == -1:
            ep = EPOCHS -1
        
        fname = 'GANResult_'+format(ep, '03d')+'.png'
        print('file='+fname)
        plt.savefig(fname)
        display(fname)

def history():
    with open('save_gimage.pkl', 'rb') as f:
        save_gimage = pkl.load(f)
        fig, axes = plt.subplots(int(EPOCHS/10), 5, figsize=(28,28))
        
        for save_gimage, ax_row in zip(save_gimage[::10], axes):
            for img, ax in zip(save_gimage[::1], ax_row):
                ax.imshow(img.reshape((28,28)), cmap='gray')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
        
        fname = 'GANHistory.png'
        print('file='+fname)
        plt.savefig(fname)
        display(fname)

def loss():
    with open('save_loss.pkl', 'rb') as f:
        save_loss = pkl.load(f)
        
        fig, ax = plt.subplots()
        loss = np.array(save_loss)
        plt.plot(loss.T[0], label='Discriminator')
        plt.plot(loss.T[1], label='Generator')
        plt.title('Loss')
        plt.legend()
        fname="GANLoss.png"
        print('file='+fname)
        plt.savefig(fname)
        display(fname)

if __name__ == '__manin__':
    args = sys.argv
    ep = 0
    
    if len(args) == 1: result(-1)
    elif args[1] == 'h': history()
    elif args[1] == 'l': loss()
    else: result(int(args[1]))
