# Code dedicated to the Sport Task MediaEval22 
__author__ = "Pierre-Etienne Martin"
__copyright__ = "Copyright (C) 2022 Pierre-Etienne Martin"
__license__ = "CC BY 4.0"
__version__ = "1.0"
import gc
import numpy as np
import torch
import random
import platform
import os
import sys
from shutil import rmtree
import matplotlib
import logging
import cv2
# To be able to save figure using screen with matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.metrics import confusion_matrix
import itertools
import math

'''
Reset Pytorch Session
'''
def reset_training(seed):
    gc.collect()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

'''
Print and log functions
'''
def print_and_log(message, log=None):
    print(message)
    if log is not None:
        log.info(message)

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    return l

def close_log(log):
    if log is not None:
        x = list(log.handlers)
        for i in x:
            log.removeHandler(i)
            i.flush()
            i.close()

'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

'''
Progression bar in the terminal
'''
def progress_bar(count, total, title, completed=0, log=None):
    terminal_size = get_terminal_size()
    percentage = int(100.0 * count / total)
    length_bar = min([max([3, terminal_size[0] - len(title) - len(str(total)) - len(str(count)) - len(str(percentage)) - 10]),20])
    filled_len = int(length_bar * count / total)
    bar = '█' * filled_len + ' ' * (length_bar - filled_len)
    # To delete the content of the current line
    sys.stdout.write('\x1b[2K')
    sys.stdout.write('%s [%s] %s %% (%d/%d)\r' % (title, bar, percentage, count, total))
    sys.stdout.flush()
    if completed:
        print_and_log('%s [%s] %s %% (%d/%d)' % (title, bar, percentage, count, total), log)


'''
Terminal size for different platform 
'''
def get_terminal_size():
    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if current_os in ['Linux', 'Darwin'] or current_os.beginswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)      # default value
    return tuple_xy

def _get_terminal_size_windows():
    try:
        from ctypes import windll, create_string_buffer
        import struct
        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        if res:
            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom,
             maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
            return sizex, sizey
    except:
        pass

def _get_terminal_size_tput():
    try:
        import subprocess, shlex
        cols = int(subprocess.check_call(shlex.split('tput cols')))
        rows = int(subprocess.check_call(shlex.split('tput lines')))
        return (cols, rows)
    except:
        pass

def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
            return cr
        except:
            pass
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (os.environ['LINES'], os.environ['COLUMNS'])
        except:
            return None
    return int(cr[1]), int(cr[0])

'''
Figures
'''
def make_train_figure(loss_train, loss_val, acc_val, acc_train, path_to_save):

    host = host_subplot(111, axes_class=AA.Axes)
    par = host.twinx()

    host.set_xlabel("Epochs")
    host.set_ylabel("Loss")
    par.set_ylabel("Accuracy")

    par.axis["right"].toggle(all=True)

    epochs = [i for i in range(1, len(loss_val)+1)]

    host.set_xlim(1, len(epochs))
    host.set_ylim(0, np.max([np.max(loss_train), np.max(loss_val)]))
    par.set_ylim(0, 1)

    max_acc = max(acc_val)
    max_acc_idx = epochs[acc_val.index(max_acc)]
    host.set_title("Max Validation Accuracy: %.1f%% at iteration %d" % (max_acc*100, max_acc_idx))

    host.plot(epochs, loss_train, label="Train loss", linewidth=1.5)
    host.plot(epochs, loss_val, label="Validation loss", linewidth=1.5)
    par.plot(epochs, acc_val, label="Validation Accuracy", linewidth=1.5)
    par.plot(epochs, acc_train, label="Train Accuracy", linewidth=1.5)

    host.legend(loc='lower right', ncol=1, fancybox=False, shadow=True)

    plt.savefig(path_to_save)
    plt.close('all')
    return True

def plot_confusion_matrix(cm, classes, save_path, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    
    acc = np.mean(np.array([cm[i,i] for i in range(len(cm))]).sum()/cm.sum()) * 100
    # Normalize the confusion matrix for colormapping
    # Transpose the matrix to divide each row of the matrix by each vector element. Transpose the result to return to the matrix’s previous orientation.
    cm = (cm.T / [max(tmp,1) for tmp in cm.sum(axis=1)]).T
    acc_2 = np.array([cm[i,i] for i in range(len(cm))])

    title = 'Accuracy of %.1f%%\n$\\mu$ = %.1f with $\\sigma$ = %.1f' % (acc, np.mean(acc_2)*100, np.std(acc_2)*100)
    if len(classes)>=12:
        plt.subplots(figsize=(12,12))
    elif len(classes)>=6:
        plt.subplots(figsize=(8,8))
    else:
        plt.subplots(figsize=(5,5))

    plt.imshow(cm.astype('float'), interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')


'''
Frame Extractor
'''
def frame_extractor(video_path, width, save_path):
    # Load Video
    cap = cv2.VideoCapture(video_path)
    length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    # Check if video uploaded
    if not cap.isOpened():
        sys.exit("Unable to open the video, check the path.\n")

    while frame_number < length_video:

        # Load video
        _, rgb = cap.read()

        # Check if load Properly
        if _ == 1:
            # Resizing and Save
            rgb = cv2.resize(rgb, (width, rgb.shape[0]*width//rgb.shape[1]))
            cv2.imwrite(os.path.join(save_path, '%08d.png' % frame_number), rgb)
            frame_number+=1
    cap.release()