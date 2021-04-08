from luigi.util import inherits

from config import DownloadGeneratorDatasetParameter, MakeGeneratorDatasetParameter
from modules import myluigi
from preprocess.download_dataset import main as download_dataset
from preprocess.make_dataset import main as make_dataset


class DownloadGeneratorDataset(DownloadGeneratorDatasetParameter, myluigi.TaskBase):
    '''
    This task downloads the dataset if dataset d == 'mnist'
    '''
    def requires(self):
        return []

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        return download_dataset(out_dir=temp_output_path, **self.params)

    def get_kwargs(self):
        kwargs = dict(
            dataset_name=self.d
        )
        return kwargs


@inherits(DownloadGeneratorDataset)
class MakeGeneratorDataset(MakeGeneratorDatasetParameter, myluigi.TaskBase):
    '''
    This task splits the dataset into 'train', 'valid', and 'test' dataset if self.d == 'mnist'.
    And it generates 2D-normal samples to construct 'train', 'valid', and 'test' dataset if self.d == 'mulvargaussian'
    The output is the numpy dumped dataset placed under sub directories: 'train', 'valid', and 'test'.
    '''
    def requires(self):
        return [self.clone(DownloadGeneratorDataset)]

    def run_within_temporary_path(self, temp_output_path):
        self.params = self.get_kwargs()
        self.params['dataset_dir'] = self.input()[0].path
        return make_dataset(out_dir=temp_output_path, **self.params)

    def get_kwargs(self):
        kwargs = dict(
            nclasses=self.nc,
            dataset_name=self.d,
            permute_train=True,
            permute_test=True,
            seed=self.seed,
            nsamp_train=self.s_tr,
            nsamp_valid=self.s_va,
            nsamp_test=self.s_te
        )
        return kwargs
