import wandb
from train import Train
from eval import Evaluation
from constants import test_path, corpus_path
from typing import Optional, Union

class HyperparameterSearch(Train):
    def __init__(self, sweep_id: str) -> None:
        """
        A class for hyperparameter search using sweep in wandb
        """
        super().__init__()
        self.sweep_id = self.wandb_args['sweep_id']
        self.times = self.wandb_args['times']
        
    def _search(self, config=None):
        """
        Initial a new run for hyper search 
        """
        with wandb.init(config=config):
            config = wandb.config
        self.model_args['distillation_loss'] = config.distillation_loss
        self.data_args['num_sample_train'] = config.num_sample_train
        self.training_args.num_train_epochs = config.num_train_epochs
        self.training_args.per_device_train_batch_size = config.per_device_train_batch_size
        self.training_args.learning_rate = config.learning_rate
        self.data_args['max_seq_length'] = config.max_seq_length
        self.model_args['mlp_only_train'] = config.mlp_only_train
        self.model_args['pooler_type'] = config.pooler_type
        self.model_args['tau2'] = config.tau2
        self.model_args['temp'] = config.temp
        self.model_args['mlm_weight'] = config.mlm_weight
        self.data_args['mlm_probability'] = config.mlm_probability
        try:
            loss = self.training_with_wandb(self.model_args, self.data_args, self.training_args)
            
            print("****** Evaluation ******")
            
            model_path = self.training_args.output_dir
            
            evaluation = Evaluation(test_path=test_path,
                                    model_path=model_path,
                                    corpus_path=corpus_path,
                                    batch_size= 128
                                    )
            result = evaluation.evaluation(k = 20)
            wandb.log({'Recall@10': result})
        except:
            print('Error when computing recall')
            raise
    
    def start_search(self, times:Optional[int] = None ) -> None:
        times = None or self.times
        """
        Start new hyperparameter search times
        Args:
            times: The number of the searches
            
        Return: 
            None
        """
        wandb.agent(sweep_id = self.sweep_id, function=self._search, count=times)
        
if __name__ == '__main__':
    hyperparameter_search = HyperparameterSearch('rankcse_minilog/dexi056r')
    hyperparameter_search.start_search()