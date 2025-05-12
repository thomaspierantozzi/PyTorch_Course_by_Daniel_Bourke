import os
import pathlib
from scripts import utilities
from random import shuffle, sample
from torchvision.datasets import Food101
import shutil


def download_Food101_dataset(path: pathlib.Path, logger: utilities.Logging_Agent = None) -> None:
    '''
    function to download food 101 dataset from pytorch libraries
    :param path: path to download dataset
    :param logger: a logging agent defined by means of the utilities script
    :return: None
    '''
    if not path.exists():
        os.makedirs(path)
        Food101(root=path, download=True)
        logger.write_log(f'The Food101 dataset has been downloaded and is available @ {path}')
        shutil.copytree(path / 'food-101/images', path / 'images')
        shutil.rmtree(path / 'food-101') #bug corrected here
        os.remove(path / 'food-101.tar.gz')
    else:
        logger.write_log(f'The Food101 dataset already exists @ {path}')
    return None

def pick_foods(food101_folder, logger: utilities.Logging_Agent, number_of_foods: int = 5, ) -> list:
    '''
    A function to pick 5 random dishes from the food101 dataset.
    :param food101_folder: path to food101 dataset
    :param number_of_foods: number of foods to pick from the dataset
    :param logger: a logging agent defined by means of the utilities script
    :return: picked_food in a list of 'number_of_foods' length
    '''

    #we pick 5 dishes amongst the ones in the original dataset
    food_list = [] #a list to append the name of the different dishes
    for (path, subdirs, files) in os.walk(food101_folder):
        food_name = path.split('/')[-1]
#        print(f'\tThe {food_name} folder has {len(subdirs)} subfolders & {len(files)} files.')
        if food_name != 'food101' and food_name != 'images' and food_name!='food-101':
            food_list.append(food_name)

    #let's take randomly five dishes to base our model on
    picked_food = sample(food_list, k=number_of_foods)
    

    logger.write_log(f'The picked food are: {picked_food}')
    return picked_food

def create_train_test_folders(picked_foods: list,
                              food101_folder: pathlib.Path,
                              train_path: pathlib.Path,
                              test_path: pathlib.Path,
                              train_percentage: float = 0.8,
                              logger: utilities.Logging_Agent = None
) -> None:
    '''
    Create two folders on the base of the picked_foods chosen to train the model on
    :param train_percentage (default 0.80)
    :param picked_foods: list of picked_foods to copy and paste into the final train and test folders
    :param food101_folder: the folder in which the original dataset has been stored
    :param train_path: the path to the train folder
    :param test_path: the path to the test folder
    :return: None
    '''
    for food in picked_foods:
        path_to_copy = food101_folder / 'images' / food
        picked_food_samples = os.listdir(path_to_copy)
        logger.write_log(f'The picked food {food} has {len(picked_food_samples)} samples.')
        shuffle(picked_food_samples)
        train_samples_number = int(train_percentage * len(picked_food_samples))
        train_samples = picked_food_samples[:train_samples_number]
        test_samples = picked_food_samples[train_samples_number:]
        os.mkdir(train_path / food)
        os.mkdir(test_path / food)
        for sample in train_samples:
            shutil.copy(path_to_copy/sample, train_path / food)
        logger.write_log(f'Copied {len(train_samples)} files to {train_path / food}')
        for sample in test_samples:
            shutil.copy(path_to_copy/sample, test_path / food)
        logger.write_log(f'Copied {len(test_samples)} files to {test_path / food}')
        shutil.copy(path_to_copy/sample, test_path / food)
    return None

def dataset_creation(food101_folder: pathlib.Path,
                     dataset_train_folder: pathlib.Path,
                     dataset_test_folder: pathlib.Path,
                     logger: utilities.Logging_Agent,
                     nr_classes: int = 5,
                     **kwargs) -> None:
    '''
    the function creates two folder (train and test) in order for torch.ImageFolder to work out succesfully and giving a proper dataset of pics
    :param food101_folder: the folder where the original Food101 dataset has been stored
    :param nr_classes: number of classes to consider for the training (default 5)
    :param dataset_train_folder: train folder path
    :param dataset_test_folder: test folder path
    :param logger: a logging agent defined by means of the utilities script
    :param kwargs: 'picked_foods' can be passed as a keyword argument. If 'picked_foods' is passed, it will pick the list of foods and create the train and test folders accordingly
    :return: None
    '''

    #check if the folders already exist. If yes then nothing will be downloaded nor created. PLEASE be sure that the content of the folder is consisten with what requested with the training
    if not (dataset_train_folder.exists() and dataset_test_folder.exists()):
        logger.write_log(f'Creating dataset train and test folders:')
        os.mkdir(dataset_train_folder)
        os.mkdir(dataset_test_folder)
        logger.write_log(f'The folders have been created:\n'
              f'\t{dataset_train_folder}'
              f'\n\t{dataset_test_folder}')

        if food101_folder.exists():
            logger.write_log('The food101 folder exist...')
        else:
            logger.write_log('The food101 folder does not exist...The download will start soon...')
            download_Food101_dataset(path=food101_folder, logger=logger)

        if 'picked_foods' in kwargs:
            picked_foods = kwargs['picked_foods']
        else:
            picked_foods = pick_foods(number_of_foods=nr_classes, food101_folder=food101_folder, logger=logger)
        create_train_test_folders(train_percentage=0.8,
                                  picked_foods=picked_foods,
                                  food101_folder=food101_folder,
                                  train_path=dataset_train_folder,
                                  test_path=dataset_test_folder,
                                  logger=logger
                                  )
    else:
        logger.write_log(f'The folders already exist: \n'
                  f'\t{dataset_train_folder}\n'
                  f'\t{dataset_test_folder}\n'
                  f'\tFoods in the train and test folders: \n\t\t{os.listdir(dataset_train_folder)}')
    return os.listdir(dataset_train_folder)

if __name__ == '__main__':
    import utilities
    os.chdir('/Users/thomaspierantozzi/PycharmProjects/PyTorch_Train/05_PyTorch_Food101')
    FOOD101_PATH = pathlib.Path(os.getcwd() + '/Datasets/food-101')  # folder where the original dataset is stored in the
    DATASET_TRAIN_FOLDER = pathlib.Path(os.getcwd() + '/Datasets/train')
    DATASET_TEST_FOLDER = pathlib.Path(os.getcwd() + '/Datasets/test')
    LOG_FOLDER = pathlib.Path(os.getcwd() + '/saved_models/new_model')
    logger = utilities.Logging_Agent(log_path=LOG_FOLDER)

    test = dataset_creation(
        food101_folder=FOOD101_PATH,
        nr_classes=5,
        dataset_train_folder=DATASET_TRAIN_FOLDER,
        dataset_test_folder=DATASET_TEST_FOLDER,
        logger=logger
    )

    print(test)