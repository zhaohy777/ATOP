import os
import random

import torch.nn.functional
from torch.utils.data import RandomSampler
from yacs.config import CfgNode
from .data_utils import InputExample, InputFeatures
from .data_sampler import FewShotSampler
from .data_processor import DataProcessor
from .lama_dataset import LAMAProcessor
from .relation_classification_dataset import TACREDProcessor, TACREVProcessor, ReTACREDProcessor, SemEvalProcessor
from .superglue_dataset import WicProcessor, RteProcessor, CbProcessor, WscProcessor, BoolQProcessor, CopaProcessor, MultiRcProcessor, RecordProcessor
from .typing_dataset import FewNERDProcessor
from .conditional_generation_dataset import WebNLGProcessor

from .typing_dataset import PROCESSORS as TYPING_PROCESSORS
from .text_classification_dataset import PROCESSORS as TC_PROCESSORS
from .superglue_dataset import PROCESSORS as SUPERGLUE_PROCESSORS
from .relation_classification_dataset import PROCESSORS as RC_PROCESSORS
from .lama_dataset import PROCESSORS as LAMA_PROCESSORS
from .conditional_generation_dataset import PROCESSORS as CG_PROCESSORS
from .lmbff_dataset import PROCESSORS as LMBFF_PROCESSORS

from openprompt.utils.logging import logger


PROCESSORS = {
    **TYPING_PROCESSORS,
    **TC_PROCESSORS,
    **SUPERGLUE_PROCESSORS,
    **RC_PROCESSORS,
    **LAMA_PROCESSORS,
    **CG_PROCESSORS,
    **LAMA_PROCESSORS,
    **LMBFF_PROCESSORS,
}


def load_dataset(config: CfgNode, return_class=True):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.
    
    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
        return_class (:obj:`bool`): Whether return the data processor class
                    for future usage.

    Returns:
        :obj:`Optional[List[InputExample]]`: The train dataset.
        :obj:`Optional[List[InputExample]]`: The valid dataset.
        :obj:`Optional[List[InputExample]]`: The test dataset.
        :obj:"
    """
    maxlen = 0
    valid_dataset = []
    test_dataset = []
    dataset_config = config.dataset
    processor = PROCESSORS[dataset_config.name.lower()]()
    try:
        if not config.dataset.domains:
            train_dataset = processor.get_train_examples(dataset_config.path)
        else:
            all_min = 1314
            train_dataset = []
            # train_dataset = processor.get_train_examples2(config)
            for domain in config.dataset.domains:
                if domain != config.dataset.target_domain:
                    # 获得源域数据
                    # train_dataset.append(((processor.get_train_examples(os.path.join(dataset_config.path, str(domain)),domain))))
                    train_dataset.append(((processor.get_train_examples(os.path.join(dataset_config.path, str(domain)),domain))*3)[:all_min])
            # 目标域数据
            train_examples = processor.get_train_examples(os.path.join(dataset_config.path, str(config.dataset.target_domain)),config.dataset.target_domain)
            train_dataset.append(((train_examples*3)[:all_min]))

        for dm in train_dataset:
            for exa in dm:
                lenth = len(exa.text_a.split(" "))
                if lenth > maxlen: maxlen = lenth

    except FileNotFoundError:
        logger.warning("Has no training dataset.")
        train_dataset = None
    try:
        ### 用源域做验证集
        # for domain in config.dataset.domains:
        #     if domain != config.dataset.target_domain:
        #         valid_dataset = valid_dataset + (
        #         (processor.get_dev_examples(os.path.join(dataset_config.path, str(domain)), domain)))
        # get_dev_examples || get_test_examples
        valid_dataset = (processor.get_dev_examples
                         (os.path.join(dataset_config.path, str(config.dataset.target_domain)),config.dataset.target_domain))

        for exa in valid_dataset:
            lenth = len(exa.text_a.split(" "))
            if lenth>maxlen: maxlen = lenth
        print(maxlen)
        # valid_dataset = processor.get_dev_examples2(config)
        # 用目标域做验证集
        # valid_dataset = valid_dataset + (
        #             processor.get_dev_examples(os.path.join(dataset_config.path, str(config.dataset.target_domain)), config.dataset.target_domain))
    except FileNotFoundError:
        logger.warning("Has no valid dataset.")
        valid_dataset = None
    try:
        test_dataset = processor.get_test_examples(os.path.join(dataset_config.path, str(config.dataset.target_domain)), config.dataset.target_domain)

        # test_dataset = test_dataset + (
        #     processor.get_test_examples(os.path.join(dataset_config.path, str(config.dataset.target_domain)), config.dataset.target_domain))
    except FileNotFoundError:
        logger.warning("Has no test dataset.")
        test_dataset = None
    # checking whether donwloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    if return_class:
        return train_dataset, valid_dataset, test_dataset, processor
    else:
        return  train_dataset, valid_dataset, test_dataset