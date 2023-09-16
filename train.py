import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertForPreTraining,
    PhobertTokenizer
)
from datasets import load_dataset, DatasetDict
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
import shutil
import wandb
import torch
import logging

from utils.fronts import Font
from constants import load_config, train_config, eval_config, segment_pyvi, output_path
from eval import Evaluation
from rankcse.trainers import CLTrainer
from rankcse.models import RobertaForCL, BertForCL

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s:[ %(levelname)s ]:\t%(message)s ')
font = Font()
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class OurDataCollatorWithPadding:

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 data_args: dict,
                 model_args: dict,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 mlm: bool = True) -> None:

        self.tokenizer = tokenizer
        self.model_args = model_args
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mlm = mlm
        self.mlm_probability = data_args['mlm_probability']

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask',
                        'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append(
                    {k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            # padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            # truncation = True
        )
        if self.model_args['do_mlm']:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(
                batch["input_ids"])

        batch = {k: batch[k].view(
            bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class CustomTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Train:
    def __init__(self) -> None:
        """
        Initial parameter from config file
        """
        self.args = load_config(train_config)
        self.model_args = self.args['model_args']
        self.data_args = self.args['data_args']
        self.training_args = CustomTrainingArguments(**self.args['training_args'])
        self.wandb_args = self.args['wandb_args']

    def _load_dataset(self) -> DatasetDict:
        """
        The function will help to read data from file and segment before training  

        Args: 
            None

        Return: 
            A DatasetDict have one columns is train
        """
        data_files = {}
        if self.data_args['train_file'] is not None:
            data_files["train"] = self.data_args['train_file']
        extension = self.data_args['train_file'].split(".")[-1]
        if extension == "txt":
            extension = "text"
        if extension == "csv":
            datasets = load_dataset(extension, 
                                    data_files=data_files, 
                                    cache_dir="./data/",
                                    delimiter="\t" if "tsv" in self.data_args['train_file'] else ",")
        else:
            datasets = load_dataset(extension, 
                                    data_files=data_files, 
                                    cache_dir="./data/")
        datasets = datasets['train'].select(range(self.data_args['num_sample_train']))
        datasets = DatasetDict({
            'train': datasets
        })
        datasets = datasets.map(segment_pyvi)
        return datasets

    def _training_initial(self):
        """
        Initial some arguments and model before training

        Args:
            None

        Return:
            A model for training and
            A toketizer corresponds to the model
        """

        config_kwargs = {
            "cache_dir": self.model_args['cache_dir'],
            "revision": self.model_args['model_revision'],
            "use_auth_token": True if self.model_args['use_auth_token'] else None,
        }
        if self.model_args['config_name']:
            config = AutoConfig.from_pretrained(
                self.model_args['config_name'], **config_kwargs)
        elif self.model_args['model_name_or_path']:
            config = AutoConfig.from_pretrained(
                self.model_args['model_name_or_path'], **config_kwargs)
        else:
            config = CONFIG_MAPPING[self.model_args['model_type']]()
            logger.warning("You are instantiating a new config instance from scratch.")

        tokenizer_kwargs = {
            "cache_dir": self.model_args['cache_dir'],
            "use_fast": self.model_args['use_fast_tokenizer'],
            "revision": self.model_args['model_revision'],
            "use_auth_token": True if self.model_args['use_auth_token'] else None,
        }
        if self.model_args['model_name_or_path']:
            if 'phobert' in self.model_args['model_name_or_path']:
                tokenizer = PhobertTokenizer.from_pretrained(self.model_args['model_name_or_path'], 
                                                             **tokenizer_kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_args['model_name_or_path'], 
                                                          **tokenizer_kwargs)
        else:
            raise ValueError(
                "You should using existing model or tokenizer from huggingface"
            )
            
        roberta_structure = 'roberta' in self.model_args['model_name_or_path'] or \
                            'phobert' in self.model_args['model_name_or_path'] or \
                            'bge' in self.model_args['model_name_or_path']         
        bert_struture = 'bert' in self.model_args['model_name_or_path']
        model_exist = self.model_args['model_name_or_path']

        if model_exist:
            if roberta_structure:
                model = RobertaForCL.from_pretrained(
                    self.model_args['model_name_or_path'],
                    from_tf=bool(
                        ".ckpt" in self.model_args['model_name_or_path']),
                    config=config,
                    cache_dir=self.model_args['cache_dir'],
                    revision=self.model_args['model_revision'],
                    use_auth_token=True if self.model_args['use_auth_token'] else None,
                    model_args=self.model_args
                )
            elif bert_struture:
                model = BertForCL.from_pretrained(
                    self.model_args['model_name_or_path'],
                    from_tf=bool(
                        ".ckpt" in self.model_args['model_name_or_path']),
                    config=config,
                    cache_dir=self.model_args['cache_dir'],
                    revision=self.model_args['model_revision'],
                    use_auth_token=True if self.model_args['use_auth_token'] else None,
                    model_args=self.model_args
                )
                if self.model_args['do_mlm']:
                    pretrained_model = BertForPreTraining.from_pretrained(
                        self.model_args['model_name_or_path'])
                    model.lm_head.load_state_dict(
                        pretrained_model.cls.predictions.state_dict())
            else:
                raise NotImplementedError
        if not model_exist:
            raise NotImplementedError

        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def _prepare_columns(self, datasets: DatasetDict) -> Tuple:
        """
        Get feature in the dataset for training

        Args: 
            A dataset that cleaned for training

        Return:
            A tuple contains all feature in data set
        """

        column_names = datasets["train"].column_names
        sent_2_cname = None
        if len(column_names) == 2:
            # Pair datasets
            sent_0_cname = column_names[0]
            sent_1_cname = column_names[1]
        elif len(column_names) == 3:
            # Pair datasets with hard negatives
            sent_0_cname = column_names[0]
            sent_1_cname = column_names[1]
            sent_2_cname = column_names[2]
        elif len(column_names) == 1:
            # Unsupervised datasets
            sent_0_cname = column_names[0]
            sent_1_cname = column_names[0]
        else:
            raise NotImplementedError

        return sent_0_cname, sent_1_cname, sent_2_cname

    def _log_param(self,
                   trainer,
                   train_result,
                   training_args: CustomTrainingArguments,
                   model_args: dict,
                   data_args: dict) -> None:
        """
        Logging all parameter and result to txt file and save state to json file

        Args:
            trainer: Trainer huggingface
            train_result: All information in training process
            training_args: Training arguments
            model_args:  Model arguments
            data_args:  Data arguments
        Return:
            None
        """
        output_train_file = os.path.join(training_args.output_dir, 
                                         "train_results.txt")
        with open(output_train_file, "w") as writer:
            writer.write(f"*****Model Arguments*****\n")
            for key, value in model_args.items():
                writer.write(f"{key.upper()}: {value}\n")

            writer.write(f"*****Data Arguments*****\n")
            for key, value in data_args.items():
                writer.write(f"{key.upper()}: {value}\n")

            writer.write(f"*****Training Arguments*****\n")
            writer.write(f"LEARNING RATE: {training_args.learning_rate}\n")
            writer.write(f"BATCH SIZE: {training_args.per_device_train_batch_size}\n")
            writer.write(f"EPOCH: {training_args.num_train_epochs}\n")
            writer.write(f"AVERAGE LOSS: {train_result.training_loss}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, 
                                                "trainer_state.json"))

    def _prepare_features(self,
                          examples,
                          sent_0_cname: str,
                          sent_1_cname: str,
                          sent_2_cname: str,
                          tokenizer,
                          data_args: dict):
        """
        If no sentence in the batch exceed the max length, then use
        the max sentence length in the batch, otherwise use the 
        max sentence length in the argument and truncate those that
        exceed the max length.
        padding = max_length (when pad_to_max_length, for pressure test)
        All sentences are padded/truncated to data_args['max_seq_length.
        """
        total = len(examples[sent_0_cname])

        # Avoid "None" fields
        for idx in range(total):
            examples[sent_0_cname][idx] = examples[sent_0_cname][idx] or " "
            examples[sent_1_cname][idx] = examples[sent_1_cname][idx] or " "

        sentences = examples[sent_0_cname] + examples[sent_1_cname]

        # If hard negative exists
        if sent_2_cname is not None:
            for idx in range(total):
                examples[sent_2_cname][idx] = examples[sent_2_cname][idx] or " "
            sentences += examples[sent_2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args['max_seq_length'],
            truncation=True,
            padding="max_length" if data_args['pad_to_max_length'] else False,
        )

        features = {}
        if sent_2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key]
                                  [i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i],
                                  sent_features[key][i+total]] for i in range(total)]
        return features

    def training(self,
                 model_args: Optional[dict] = None,
                 data_args: Optional[dict] = None,
                 training_args: Optional[CustomTrainingArguments] = None):
        """
        Training model rankcse

        Args:
            training_args: Training arguments
            model_args:  Model arguments
            data_args:  Data arguments

        Return:
            The average loss from training process and
            The model
        """
        not_init_arg = not (model_args and data_args and training_args)
        if not_init_arg:
            model_args = self.model_args
            data_args = self.data_args
            training_args = self.training_args

        output_exist_and_not_empty = os.path.exists(training_args.output_dir) \
                                    and os.listdir(training_args.output_dir) \
                                    and training_args.do_train \
                                    and not training_args.overwrite_output_dir
        if output_exist_and_not_empty:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        logger.info("Training/evaluation parameters %s", training_args)

        # Set seed before initializing model.
        set_seed(training_args.seed)
        datasets = self._load_dataset()
        model, tokenizer = self._training_initial()
        # Prepare features
        sent_0_cname, sent_1_cname, sent_2_cname = self._prepare_columns(datasets)
        column_names = datasets["train"].column_names

        if training_args.do_train:
            train_dataset = datasets["train"].map(
                self._prepare_features,
                batched=True,
                num_proc=data_args['preprocessing_num_workers'],
                remove_columns=column_names,
                load_from_cache_file=not data_args['overwrite_cache'],
                fn_kwargs={
                    "sent_0_cname": sent_0_cname,
                    "sent_1_cname": sent_1_cname,
                    "sent_2_cname": sent_2_cname,
                    "tokenizer": tokenizer,
                    "data_args": data_args}
            )
        
        if data_args['pad_to_max_length']:
            data_collator = default_data_collator
        else:
            data_collator = OurDataCollatorWithPadding(tokenizer=tokenizer,
                                                       model_args=self.model_args,
                                                       data_args=self.data_args)

        training_args.first_teacher_name_or_path = model_args['first_teacher_name_or_path']
        training_args.second_teacher_name_or_path = model_args['second_teacher_name_or_path']
        training_args.tau2 = model_args['tau2']
        training_args.alpha_ = model_args['alpha_']

        trainer = CLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.model_args = model_args

        model_valid = model_args['model_name_or_path'] is not None \
                      and os.path.isdir(model_args['model_name_or_path'])
        # Training
        if training_args.do_train:
            if model_valid:
                model_path = model_args['model_name_or_path']
            else: 
                model_path = None
            train_result = trainer.train(model_path=model_path)
            trainer.save_model()  # Saves the tokenizer too for easy upload

        self._log_param(trainer=trainer,
                        train_result=train_result,
                        training_args=training_args,
                        model_args=model_args,
                        data_args=data_args
                        )
        return train_result.training_loss, model

    def training_with_wandb(self, model_args = None, data_args = None, training_args = None):
        model_args = model_args or self.model_args
        data_args = data_args or self.data_args
        training_args = training_args or self.training_args

        print('************', self.wandb_args['project'])
        """
        Training model and then log hyperparameter to wandb
        Args:
            None
        
        Return:
            None
        """
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        loss, model = self.training()
        eval_args = load_config(eval_config)
        eval_args['model_path'] = self.training_args.output_dir
        evaluation = Evaluation(args=eval_args)
        eval_result = {}
        for k in set(self.wandb_args['log_k']):
            eval_result[f"recall@{k}"] = evaluation.evaluation(k=k)

        run = wandb.init(
            project=self.wandb_args['project'],
            group="research",
            entity="simcse",
            name='fine-tuning-test')
        wandb.watch(model, log=None)

        metadata = {
            "model_name_or_path": self.model_args["model_name_or_path"],
            "first_teacher_name_or_path": self.model_args["first_teacher_name_or_path"],
            "second_teacher_name_or_path": self.model_args["second_teacher_name_or_path"],
            "distillation_loss": self.model_args["distillation_loss"],
            "tau2": self.model_args["tau2"],
            "alpha_": self.model_args["alpha_"],
            "beta_": self.model_args["beta_"],
            "gamma_": self.model_args["gamma_"],
            "temp": self.model_args["temp"],
            "pooler_type": self.model_args["pooler_type"],
            "mlm_weight": self.model_args["mlm_weight"],
            "mlp_only_train": self.model_args["mlp_only_train"],
            "num_sample_train": self.data_args["num_sample_train"],
            "max_seq_length": self.data_args["max_seq_length"],
            "train_file": self.data_args["train_file"],
            "pad_to_max_length": self.data_args["pad_to_max_length"],
            "mlm_probability": self.data_args["mlm_probability"],
            "num_train_epochs": self.training_args.num_train_epochs,
            "per_device_train_batch_size": self.training_args.per_device_train_batch_size,
            "learning_rate": self.training_args.learning_rate,
            "fp16": self.training_args.fp16,
            "loss": loss,
            **eval_result
        }

        new_model = wandb.Artifact(name=f"model_{run.id}",
                                   type='model',
                                   description=self.wandb_args['description'],
                                   metadata=metadata)
        # new_model.add_dir(self.training_args.output_dir,
        #                   name=f"model_{run.id}")
        
        run.log_artifact(artifact_or_path=new_model,
                         name='linh',
                         type='model')
        run.link_artifact(new_model, 
                          self.wandb_args['model_registry'], 
                          aliases='')
        run.finish()
        logging.info(font.info_text(
            f"Save model registry to wandb successfully"))

if __name__ == "__main__":
    train = Train()
    train.training_with_wandb()
    # train.training()
