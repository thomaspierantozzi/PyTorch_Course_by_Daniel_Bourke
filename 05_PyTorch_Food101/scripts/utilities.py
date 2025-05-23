
'''
This script is a container for different functions which are only aiding the core of the programs written
'''

import pathlib
import requests
import datetime
import os
import matplotlib.pyplot as plt
from random import choice
from PIL import Image
from torch import nn
import pickle
import torch
import pathlib
import random

def send_telegram(message: str):
    '''
    this function sends message to telegram via a chat whose id must be stored in the environmental variables under the name 'chat_id'. The same applies to the token whic must be stored in the environmental variable and named 'TOKEN'
    :param message: the string message which must be sent to the telegram bot
    :return: 
    '''
    
    #check that the message is of the correct type
    if not (type(message) is str):
        raise TypeError('send_telegram function: Message must be of type str')
    
    #check the presence of the proper env variables 
    try:
        TOKEN = os.environ.get('TOKEN')
        chat_id = os.environ.get('chat_id')
    except Exception as e:
        print('send_telegram function: TOKEN and chat_id must be stored in the environmental variables (*.env file)')
    
    #tries to send a message. if the communication cannot be established then raises an Exception but doesn't block the execution of the main program    
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
        is_sent = False
        while is_sent == False:
            telegram_response = requests.get(url).json()
            if telegram_response['ok'] == True:
                is_sent = True
    except Exception as e:
        print('The message has not been sent to telegram bot, due to the following Exception: \n\t' + str(e))
        print_log(f'Error in send_telegram:\n\t {e}')

def create_model_folder(model_folder: pathlib.Path):
    '''
    check whether the folder already exists and asks if an overwrite is needed. If yes, then deletes the existing folder
    '''
    
    if type(model_folder) is str:
        model_folder = pathlib.Path(model_folder)
    
    if not model_folder.exists():
        os.mkdir(model_folder)
        print_log('Model folder created!')
    else:
        overwrite = input('Model folder already exists, do you want to overwrite it? (y/n)')
        if overwrite == 'y':
            shutil.rmtree(model_folder)
            os.mkdir(model_folder)
        else:
            print_log(f'Using the previously created model folder...')

class Logging_Agent():
    __instance=None
    def __new__(cls, log_path:pathlib.Path):
        '''
        Instantiate a singleton
        :param log_path: the path to the log file
        '''
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance
    
    def __init__(self, log_path: pathlib.Path):
        '''
        The LoggigAgent must be instantiated with a path to save the log file to
        :param log_path: the path to save the log file to
        '''
        if log_path.exists():
            self.path = log_path
        else:
            log_path.mkdir(exist_ok=True, parents=True)
            self.path = log_path
        
    def write_log(self, message: str):
        '''
        write a log file to the log.txt file, in the self.path folder 
        :param message: the message which must be logged
        :return: 
        '''
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.path / 'log.txt', 'a') as log_file:
            log_file.write(f'[{now_str:<20}]\n{message}\n')
        print(message)

    def __repr__(self):
        return f'Logging Agent pointing to: {self.path}'

def plot_samples(
        path: pathlib.Path,
        classes_names: list,
) -> None:
    '''
    Function to plot 9 samples from the train folder
    :param path: path to the train folder (or to any given folder containing samples of pictures)
    :param classes_names: names of the folder contained in the path folder. Generally this list is a list of the classes names
    :return: None
    '''
    pics_list = []
    for food in classes_names:
        for file in os.listdir(path / food):
            pics_list.append(path/food/file)

    TILES = 9

    fig = plt.figure(figsize=(8, 8))
    for index, tile in enumerate(range(TILES)):
        ax = fig.add_subplot(3, 3, index + 1)
        ax.axis('off')
        random_pic = choice(pics_list)
        with Image.open(random_pic) as pic:
            ax.imshow(pic)
            ax.set_title(f'{random_pic.parent.name}')
    return None

def plot_samples_transformed(
        path: pathlib.Path,
        classes_names: list,
        transform: nn.Sequential,
) -> None:
    '''
    Function to plot 9 samples from the train folder
    :param transform: transform pipeline set by means of a nn.Sequential class
    :param path: path to the train folder (or to any given folder containing samples of pictures)
    :param classes_names: names of the folder contained in the path folder. Generally this list is a list of the classes names
    :return: None
    '''
    pics_list = []
    for food in classes_names:
        for file in os.listdir(path / food):
            pics_list.append(path/food/file)

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    for index, tile in enumerate(range(6)):
        rand_pic = choice(pics_list)
        with Image.open(rand_pic) as pic:
            ax_base = fig.add_subplot(6, 2, 2 * index + 1)
            ax_base.imshow(pic)
            ax_base.set_title(f'Base image: {rand_pic.parent.name}')
            ax_trans = fig.add_subplot(6, 2, 2 * index + 2)
            ax_trans.imshow(transform(pic).permute(1, 2, 0))
            ax_trans.set_title(f'Transformed image: {rand_pic.parent.name}')
            ax_base.set_axis_off()
            ax_trans.set_axis_off()
    return None

def save_model(path: pathlib.Path,
               model: nn.Module,
               optimizer: torch.optim,
               epoch_nr: int,
               learning_rate: float,
               batch_size: int,
               nr_classes: int) -> None:
    '''
    this function saves the model and optimizer state_dict as well as the history of the metrics
    :param path: path to save the data to
    :param model: model to be saved. The model must have a 'history' attribute (e.g.: the Class Model_Blueprint)
    :param optimizer: optimizer used fro the training and for which the state_dict must be saved to resume the training itself
    :param epoch_nr: epoch number of the model to be saved
    :return: None
    '''
    if not path.exists():
        print('Creating the folder to save the model')
        os.mkdir(path)

    with open(path / f'HISTORY_{model.name}_LR{learning_rate}_BS{batch_size}_Classes{nr_classes}.pkl',
              'wb') as f:
        pickle.dump(model.history, f)
        print(f'Model metrics history saved in :\n\t{path}')

    model_checkpoint = path / f'model_checkpoint_epoch_{epoch_nr}.pt'
    torch.save({
        'epoch': epoch_nr,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_checkpoint)

    return None

def create_train_cv_from_folder(train_cv_perc: float,
                                root: str | pathlib.Path,
                                train_folder: str | pathlib.Path,
                                cv_folder: str | pathlib.Path):
    '''
    This function serves the user splitting a folder already shaped in the fashion of DatasetFolder from torchvision 
    (here the link: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.DatasetFolder.html).
    The result will be a new folder containing two subfolders: train and cv, each containing all the classes split in 
    folder. The train will get 'train_cv_perc' * len(samples for that class) samples
    :param train_cv_perc: a float number giving the percentage of the train_cv split
    :param root: the folder from which getting the data and the structure
    :param train_folder: accepting both string or pathlib.Path obj. That's the final train folder
    :param cv_folder: accepting both string or pathlib.Path obj. That's the final cv folder
    :return: None
    '''

    for path, dirname, filename in os.walk(root):
        train_indexes = []  #let's initialize an empty list of indexes for the train path of the current species.
        tmp_path = pathlib.Path(path)
        print('Working on ' + str(tmp_path.name), end=' | ')
        print('Moving', len(filename), 'pictures')
        train_indexes = random.sample(range(len(filename)), int(train_cv_perc * len(filename)))  #let's sample 'TRAIN_SPLIT_PERC' pictures, and move them to the train folder. the rest goes in the cv folder

        for index_file, file in enumerate(filename):
            tmp_filepath = pathlib.Path(tmp_path / file)

            if index_file in train_indexes:  #the sampled indexes go to train
                (train_folder / tmp_path.name).mkdir(exist_ok=True)
                tmp_filepath.rename(
                    train_folder / tmp_path.name / f'{index_file:0>3}_{tmp_path.name}{tmp_filepath.suffix}')
            else:  #the indexes not samples go to cv folder
                (cv_folder / tmp_path.name).mkdir(exist_ok=True)
                tmp_filepath.rename(
                    cv_folder / tmp_path.name / f'{index_file:0>3}_{tmp_path.name}{tmp_filepath.suffix}')
