import time
import torchvision
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from numpy import mean
from PIL import Image

class Model_Blueprint(nn.Module, ABC):
    '''
    defines an abstract class for the creation of the models. The two abstractmethods 'define_architecture' and 'forward' are to be defined in the subsequent classes, in order for them to work
    '''

    def __init__(self, name):
        '''
        This init launches the instance of a new model_blueprint, the name is needed to give the model an identity.
        :param name:
        '''
        super().__init__()
        self._outcome = dict()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'cv_loss': [],
            'cv_acc': []
        }
        self.name = name

    @abstractmethod
    def define_architecture(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    def compute_outcome(self, train_dataloader: torch.utils.data.DataLoader,
                        test_dataloader: torch.utils.data.DataLoader, loss_foo: torch.nn) -> dict:
        '''
        This function serves a trained model giving out a dictionary with the main metrics computed on the training and cv datasets
        :param train_dataloader: the train dataloader used to train the model
        :param test_dataloader: the cv / test dataloader used to train the model
        :param loss_foo: loss function used to train the model
        :return: dictionary of the metrics computed on the training and cv datasets
        '''
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []

        for X_batch, y_batch in train_dataloader:
            train_loss_batch, train_acc_batch = self.eval_step(X_batch, y_batch, self, loss_foo)
            train_losses.append(train_loss_batch)
            train_acc.append(train_acc_batch)

        for X_batch_cv, y_batch_cv in test_dataloader:
            test_loss_batch, test_acc_batch = self.eval_step(X_batch_cv, y_batch_cv, self, loss_foo)
            test_losses.append(test_loss_batch)
            test_acc.append(test_acc_batch)

        self._outcome['train_losses'] = np.mean(train_losses).item()
        self._outcome['train_acc'] = np.mean(train_acc).item()
        self._outcome['test_losses'] = np.mean(test_losses).item()
        self._outcome['test_acc'] = np.mean(test_acc).item()

    def history_update(self, metric: str, value: list) -> None:
        '''
        This function helps the user to keep track of the metrics during training of the model
        :param metric: string describing the metric to be appended to the history
        :param value: the value to be appended to the history
        :return: dictionary of the history of the metrics
        '''
        self.history[metric].append(value)
        return self._outcome

    def import_history(self, history_dict: dict) -> None:
        '''
        this function helps the user importing an history of metrics of a pre-trained model, when this model is loaded from a folder and an history has been saved with the proper format
        :param history_dict: dictionary of the history of the metrics
        :return: None
        '''
        self.history = history_dict
        return None

    @property
    def outcome(self):
        '''
        This function gives out the resume of the metrics of the trained model.
        :return: dictionary of the metrics computed on the training and cv datasets
        '''
        return self._outcome

    def accuracy(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        '''
        this helper function computes the accuracy of the model based on the batches entered
        :param prediction: output of the model
        :param target: target of the computation
        :return: accuracy of the model as a float
        '''

        max_logit = torch.argmax(prediction, dim=1)
        ok_pred_nr = torch.where(max_logit - target == 0, 1, 0).sum()
        return ok_pred_nr.item() / prediction.shape[0]

    def train_step(self,
                   batch_X: torch.Tensor,
                   batch_y: torch.Tensor,
                   optimizer: torch.optim,
                   loss_foo: torch.nn,
                   device):
        '''
        this function is used to perform a train step for the model on a single batch of the dataset.
        The function passes through the following steps:
        - forward step
        - loss computation step
        - backpropagation step
        - update of the weights through the optimizer
        :param batch_X: training batch
        :param batch_y: target labels for the training batch given in 'batch_X'
        :param optimizer: optimizer used to train the model
        :param loss_foo: loss function used to train the model
        :param device: device where the model and the batches are located for the computation
        :return: loss computed for the batch in the forward pass, accuracy of the model computed for the batch in the forward pass
        '''

        # 0. sending the batches to the correct device
        batch_X, batch_y = batch_X.to(device=device), batch_y.to(device=device)

        # 1. forward
        y_hat = self.forward(batch_X)

        # 2. loss_computation
        loss_batch = loss_foo(y_hat, batch_y)
        acc_batch = self.accuracy(y_hat, batch_y)

        # 3. backpropagation
        loss_batch.backward()

        # 4. Optimizer zero_grad and then step
        optimizer.step()
        optimizer.zero_grad()

        return loss_batch.item(), acc_batch

    def eval_step(
            self,
            batch_test_X: torch.Tensor,
            batch_test_y: torch.Tensor,
            loss_foo: torch.nn,
            device: torch.device,
            **kwargs
    ):
        '''
        this function is used to evaluate the model on a batch of the dataset.
        it performs the following steps:
        - switches on the torch.no_grad()
        - forward step
        - loss and accuracy computation step
        :param batch_test_X: batch to evaluate the model on
        :param batch_test_y: target labels for the batch to evaluate the model on
        :param loss_foo: loss function to evaluate the model
        :param device: device where the model and the batches are located for the computation
        :param kwargs:
            * return_pred: whether to return the prediction
        :return: loss and accuracy for the batch defined in the arguments
        '''

        # 5. CrossValidation step
        with torch.no_grad():
            # 0. Sending the batches to the correct device
            batch_test_X, batch_test_y = batch_test_X.to(device=device), batch_test_y.to(device=device)

            # 1. forward
            y_hat_cv = self.forward(batch_test_X)

            # 2. loss_computation
            loss_batch_cv = loss_foo(y_hat_cv, batch_test_y)
            acc_batch_cv = self.accuracy(y_hat_cv, batch_test_y)

            if 'return_pred' in kwargs:
                prediction = torch.argmax(nn.functional.softmax(y_hat_cv, dim=1), dim=1)
                return loss_batch_cv.item(), acc_batch_cv, prediction

            return loss_batch_cv.item(), acc_batch_cv

    def predict_single(self,
                       input: Image.Image | Tensor,
                       test_mode: bool = False,
                       target_value: int | torch.Tensor | np.ndarray = None,
                       ) -> dict:
        '''
        This function is used to make a prediction on a SINGLE IMAGE (either a torch.Tensor or PIL.Image.Image).
        This function runs on the cpu: for this model the goal is to run (hypothetically) on all the devices or on
        browser apps, where a gpu generally is not available
        :param input: image to be predicted. Either a single PIL.Image or a torch.Tensor is expected.
        :param test_mode: test mode 'True' means that the users wants to use the prediction to check whether the
        prediction matches an expected target value. Default value 'False'
        :param target_value: if in test_mode then the target_value is the single int value
        (or torch.Tensor or np.ndarray) which holds the expected target value for the input
        :return: dictionary containing the predicted class, prediction for the different classes,time spent for the
        prediction
        '''

        assert isinstance(input, torch.Tensor) or isinstance(input, Image.Image), ('The picture should be either a single PIL.Image '
                                                                     'or a torch.Tensor')

        #setting the device since we do not need to leverage the cpu: for this model the goal is to run (hypothetically)
        #on all the devices or on browser apps, where a gpu generally is not available
        device = 'cpu'
        self.to(device=device)
        if type(input) == torch.Tensor:
            input.to(device=device)

        self.eval()
        with torch.no_grad():
            start_time = time.time() #starting a timer to time the prediction pipeline
            try:
                input = self.model_transform(input)
            except AttributeError as e:
                print(f'The model needs to know which transformation pipeline has been used to pre-process the pictures.\n'
                      f'Please leverage the self.model_transform property...')

            if input.dim() < 4:
                input = input.unsqueeze(0)
            pred_logits = self.forward(input)
            pred_proba = nn.functional.softmax(pred_logits, dim=1)
            pred_class = torch.argmax(pred_proba, dim=1)
            end_time = time.time() #shutting off the timer
        probabilities_per_class = {idx: pred_proba[0, self.classes_map[idx]].item() for idx in (self.classes_map.keys())}
        output_dict = {
            'Predicted_class': pred_class.detach().to(device='cpu').item(),
            'Prediction_proba': probabilities_per_class,
            'Prediction_time': end_time - start_time
        }
        if test_mode:
            output_dict['Expected_Class'] = target_value
        return output_dict

    def write_epoch_results_class(self,
                                curr_iteration:int,
                                last_iteration:int,
                                train_loss:list,
                                train_acc:list,
                                cv_loss:list,
                                cv_acc:list,
                                end_time_iteration,
                                start_time_iteration,
                                ) -> str:
        '''
        this function returns a string which resumes the main stats of a trained model when a cv dataset is available.
        this function is supposed to be supposed for classifications, where both the accuracy and loss metrics are computed
        :param curr_iteration: the current iteration number
        :param last_iteration: the number of the overall iterations to be performed
        :param train_loss: the loss computed for the training batch. A list containing the loss computed for the different iterations.
        :param train_acc: the accuracy computed for the training batch. A list containing the loss computed for the different iterations.
        :param cv_loss: the loss computed for the cv batch. A list containing the loss computed for the different iterations.
        :param cv_acc: the accuracy computed for the cv batch. A list containing the loss computed for the different iterations.
        :param end_time_iteration: when the iteration stopped being timed (generally time.time obj)
        :param start_time_iteration: when the iteration stopped being timed (generally time.time obj)
        :return: a string with the overall stats for the iteration
        '''

        return (f'Epoch number: {curr_iteration + 1} out of {last_iteration}\n\t'
               f'Train loss: {mean(train_loss):.3f} | Train Accuracy: {mean(train_acc):.3%}\n\t'
               f'CV loss: {mean(cv_loss):>8.3f} | CV Accuracy:{mean(cv_acc):>11.3%}\n\t'
               f'Time taken: {end_time_iteration - start_time_iteration:.2f} seconds')

    def write_minibatch_results_class(self,
                                      batch_index:int,
                                      batch_quantity:int,
                                      train_loss:list,
                                      batch_loss_train:list,
                                      train_acc:list,
                                      batch_acc_train:list,
                                      start_time_iteration:int,
                                      end_time_iteration:int
                                      ) -> str:
        '''
        this function returns a string which resumes the main stats of a trained model when a cv dataset is available, on the single mini-batch.
        this function is supposed to be supposed for classifications, where both the accuracy and loss metrics are computed
        :param batch_index: the index of the mini-batch
        :param batch_quantity: the overall quantity of mini-batches
        :param train_loss: a list containing the history of the train loss computed to that moment
        :param batch_loss_train: the last value of the train loss computed on the last mini-batch
        :param train_acc: a list containing the history of the train accuracy computed to that moment
        :param batch_acc_train: the last value of the train accuracy computed on the last mini-batch
        :param start_time_iteration: mini-batch computation, starting time
        :param end_time_iteration: mini-batch computation, end time
        :return: a string with the overall stats for the last mini-batch
        '''
        return (f'Intermediate results for batch {batch_index:0>3} out of {batch_quantity:0>3}: '
                f'Train Loss epoch: {np.mean(train_loss):>8.3f} (last: {batch_loss_train:>8.3f}) | '
                f'Train Acc. epoch: {np.mean(train_acc):>6.2%} (last: {batch_acc_train:>6.2%}) | '
                f'Elapsed Time: {end_time_iteration - start_time_iteration:.1f} sec.')

    @property
    def model_transform(self):
        return self._model_transform

    @model_transform.setter
    def model_transform(self, transform: nn.Sequential | torchvision.transforms.Compose) -> None:
        '''
        this setter sets the transformation pipeline to preprocess a picture and get it ready for inference.
        :param transform: the preprocessing pipeline. Since this is going to be used for the inference as well, it's
        better to keep out any image augmentation pipeline out of this parameters
        :return: None
        '''
        self._model_transform = transform

    @property
    def classes_map(self) -> dict:
        return self._classes_map

    @classes_map.setter
    def classes_map(self, classes_map: dict) -> None:
        '''
        This setter sets dict which maps the classificiation classes labels to the index used at training time
        :param classes_map: a dict holding the mapping of classification classes
        :return: None
        '''
        self._classes_map = classes_map

class TinyVGG(Model_Blueprint):
    '''
    this class defines the abstract class 'Model_Blueprint' giving it the form of a TinyVGG model
    '''

    def __init__(
            self,
            name: str,
            input_size: int,
            first_hidden_channels: int,
            second_hidden_channels: int,
            output_size: int,
            first_hidden_linear_nodes: int
    ):
        '''
        this class builds a model like the following:
        - 1st layer: conv2d / relu / max_pooling with 'first_hidden_channels' channels
        - 2nd layer: conv2d / relu / max_pooling with 'second_hidden_channels' channels
        - 3rd layer: flatten / linear with 'output_size' channels as an output
        :param name: name of the model
        :param input_size: size (in channels) for the input
        :param first_hidden_channels: channels which are expected to come out from the first conv block
        :param second_hidden_channels: channels which are expected to come out from the second conv block
        :param output_size: size of the output (supposes to be columns of the output array where the rows are the different samples)
        :param first_hidden_linear_nodes: number of inputs for the very first linear layer of the fully connected block
        '''
        super().__init__(name=name)
        self.define_architecture(
            input_size,
            first_hidden_channels,
            second_hidden_channels,
            output_size,
            first_hidden_linear_nodes,
        )

    def define_architecture(
            self,
            input_size: int,
            first_hidden_channels: int,
            second_hidden_channels: int,
            output_size: int,
            first_hidden_linear_nodes: int,
    ):
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=first_hidden_channels, padding=1, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=first_hidden_channels, out_channels=first_hidden_channels, padding=1, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels=first_hidden_channels, out_channels=second_hidden_channels, padding=1, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=second_hidden_channels, out_channels=second_hidden_channels, padding=1, stride=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=first_hidden_linear_nodes, out_features=output_size),
        )

    def forward(self, x):
        return self.classifier(self.second_conv(self.first_conv(x)))

class EffNetB0(Model_Blueprint):
    '''
    This class is specifally made to import an EfficientNetB0 model as per PyTorch:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.EfficientNet_B0_Weights
    '''

    def __init__(self, name:str):
        super().__init__(name=name)
        self.define_architecture()

    def define_architecture(self):
        __weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        __pre_trained_model = torchvision.models.efficientnet_b0(weights=__weights)
        self.transformers = __weights.transforms
        self.layers = dict()
        self.architecture = nn.ModuleList([])
        for name, module in __pre_trained_model.named_modules():
            self.__dict__[name] = module
            self.architecture.append(module)
        del __pre_trained_model, __weights

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        #return self.classifier(self.avgpool(self.features(x)))

class EffNetB2(Model_Blueprint):
    '''
    This class is specifally made to import an EfficientNetB2 model as per PyTorch:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.EfficientNet_B2_Weights
    '''

    def __init__(self, name:str):
        super().__init__(name=name)
        self.define_architecture()

    def define_architecture(self):
        __weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        __pre_trained_model = torchvision.models.efficientnet_b2(weights=__weights)
        self.transformers = __weights.transforms
        self.layers = dict()
        self.architecture = nn.ModuleList([])
        for name, module in __pre_trained_model.named_modules():
            self.__dict__[name] = module
            self.architecture.append(module)
        del __pre_trained_model, __weights

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        #return self.classifier(self.avgpool(self.features(x)))

class EffNetB3(Model_Blueprint):
    '''
    This class is specifally made to import an EfficientNetB0 model as per PyTorch:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.EfficientNet_B0_Weights
    '''

    def __init__(self, name:str):
        super().__init__(name=name)
        self.define_architecture()

    def define_architecture(self):
        __weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
        __pre_trained_model = torchvision.models.efficientnet_b3(weights=__weights)
        self.transformers = __weights.transforms
        self.layers = dict()
        self.architecture = nn.ModuleList([])
        for name, module in __pre_trained_model.named_modules():
            self.__dict__[name] = module
            self.architecture.append(module)
        del __pre_trained_model, __weights

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        #return self.classifier(self.avgpool(self.features(x)))

class EffNetB4(Model_Blueprint):
    '''
    This class is specifally made to import an EfficientNetB4 model as per PyTorch:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.EfficientNet_B4_Weights
    '''

    def __init__(self, name:str):
        super().__init__(name=name)
        self.define_architecture()

    def define_architecture(self):
        __weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        __pre_trained_model = torchvision.models.efficientnet_b4(weights=__weights)
        self.transformers = __weights.transforms
        self.layers = dict()
        self.architecture = nn.ModuleList([])
        for name, module in __pre_trained_model.named_modules():
            self.__dict__[name] = module
            self.architecture.append(module)
        del __pre_trained_model, __weights

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        #return self.classifier(self.avgpool(self.features(x)))

class ViT_B_16(Model_Blueprint):
    '''
    This class imports the ViT-B_16 model as per PyTorch following link:
    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
    '''

    def __init__(self, name:str, SWAG_weights: bool = False):
        super().__init__(name=name)
        self.define_architecture()
        self.weights = SWAG_weights

    def define_architecture(self):
        if self.weights == False:
            __weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        else:
            __weights = torchvision.models.IMAGENET1K_SWAG_E2E_V1
        self.pre_trained_model = torchvision.models.vit_b_16(weights=__weights, progress=True)
        self.architecture = nn.ModuleList([])
        self.transformers = __weights.transforms

    def forward(self, x):
        return self.pre_trained_model(x)


if __name__ == '__main__':
    from PIL import Image
    image = Image.open(
    fp='/Users/thomaspierantozzi/PycharmProjects/PyTorch_Train/08_Model_Deployment/Datasets/final_test/cup_cakes/cup_cakes005.jpg'
    )
    vit_model = ViT_B_16('ViT-B_16')
