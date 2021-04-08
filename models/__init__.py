from .NNBase import NNBase
from .Generator import Generator
from .GANBase import GANBase
from .MNISTGANBase import MNISTGANBase
from ._2DGANBase import _2DGANBase
from .Classifier import Classifier
from .SmallCNNGAN import Model as SmallCNNGAN
from .CNNMNIST import Model as CNNMNIST
from .SmallMulVarGaussGAN import Model as SmallMulVarGaussGAN

models_dict = {
    SmallCNNGAN.name: SmallCNNGAN,
    CNNMNIST.name: CNNMNIST,
    SmallMulVarGaussGAN.name: SmallMulVarGaussGAN,
}