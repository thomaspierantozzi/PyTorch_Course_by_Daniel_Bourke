# Exercise Book for the PyTorch for Deeplearning Bootcamp
## Abstract
This repository is a container for all the studies and exercises carried out by me to follow along the course 'PyTorch for Deeplearning Bootcamp' held on Udemy by mr Daniel Bourke @mrdbourke

## First exercises
The notebooks related to the first lessons are grouped together since the topics were not challenging and the first folder can be considered more as a bunch of scribbles than nothing more

## 04_PyTorch_Class_Problem
This folder contains the very first attempts of building a PyTorch model for classification. Here things started to be a bit more serious and, at least, we can start seeing how a first rough ML workflow should like .

## 05_PyTorch_Food101
From this folder we start working consistently on the dataset chosen by the teacher to go deeper through the core topics of the course. 
In this specific folder we mainly see:
* how to download a dataset via code
* how to set dataset and dataloader objects in PyTorch (<a href='https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html'>link</a>)
* how to train and cross-validate a model
* how to plot the history of the training to check whether the model is underfitting /overfitting or somehow bugged

extras: here I created a folder to collect some helper scripts, which will be used to build easily the models. 
The main highlight of this folder could be considered the following class, used as a Blueprint for all the models created ever since. 
Having an abstract class helps us initializing models with given functions, attributes and properties avoiding us to write bunch of code every time from scratch


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
        
## 06_PyTorch_TransferLearning
In this section we deep dive in the transfer learning, giving a look at how leveraging trained models can boost the result of a ML based algorithm. More specifically the EfficientNet models will be instantiated and then their heads retrained on the given dataset
<a href='https://docs.pytorch.org/vision/main/models/efficientnet.html'>link to PyTorch EffNet page</a>

## 07_PyTorch_PaperReplicating
This is all about replicating the paper **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** which can be found <a href='https://arxiv.org/pdf/2010.11929'>here at this Arkix link</a>.
At first, we replicated from scratch the VisualTransformer model to find out that training such a huge model on a tiny dataset cannot work (surprise, surprise!!). We finally instantiate a pretrained model, leveraging the ModelBluePrint abstract class here above mentioned, to close the chapter seeing that basically the ViT model performs well (ca. 91% of accuracy on the CV set).
The good results, though, doesn't outperform the results on the same dataset achieved by the EffNet model which involves roughty one tenth of the parameters. 

## 08_PyTorch_ModelDeployment
The last section basically wraps up everything done so far: a capstone project through which we explore the power of tools like Gradio ([Link to Gradio doc...](https://www.gradio.app/docs)).
The project is based on comparing the performance of the models set in chapters 06 and 07, where an EffNet_B2 and a ViT have been trained on the specific dataset of the food classifier.
Here under I report a picture of the final comparison, as it is self-explicable: surprisingly I found out that for a single picture inference the ViT transformer is not only better in terms of performance (we already knew it), but even faster since it seems that its architecture helps it being more efficient. 
Even on CPUs the ViT performs faster than the EffNet_B2, that's why I chose it for the web app on Gradio, even though the model file weights more than 300MB. 
For a real deployment that should be taken in consideration

![Accuracy vs. InferenceTime per single image](https://github.com/thomaspierantozzi/PyTorch_Course_by_Daniel_Bourke/raw/main/08_Model_Deployment/Accuracy_vs_inference_time.png "Accuracy vs. InferenceTime")


## Next steps...
For sure this course gave me a lot, but it's obviously a starting point for something completely new. 
The next step for sure will be finding a perimeter where I can replicate what's been done through the course, as reiterating is a good way to carve this workflow into my mind and be ready for a future as a professional. 
Probably, I will try to build a web app to classify birds picture took in Europe. I need oknly to find a reliable dataset to start working on that project.

Thanks for having passed by...