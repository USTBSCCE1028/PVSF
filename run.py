import csv
import os
import argparse
import logging
import sys

from tqdm import tqdm

sys.path.append("..")
print(sys.path)
import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from model.bert_model import NERModel, EEModel, MEModel
from processor.dataset import EEProcessor, NERProcessor, EEDataset, NERDataset, MEProcessor, MEDataset
from modules.train import EETrainer, NERTrainer, METrainer
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'EE': EEModel,
    'NER': NERModel,
    'ME': MEModel
}

TRAINER_CLASSES = {
    'EE': EETrainer,
    'NER': NERTrainer,
    'ME': METrainer
}
DATA_PROCESS = {
    'EE': (EEProcessor, EEDataset),
    'NER': (NERProcessor, NERDataset),
    'ME':(MEProcessor, MEDataset)
}

DATA_PATH = {
    'MEE': {
            'train': 'data/train.json',
            'dev': 'data/val.json',
            'test': 'data/test.json',
            're_path': 'data/Event_type.json',
            'auximgs': 'data/auximgs_path.txt',
            'prompts': 'data/prompts.csv',
            'description': 'data/description.csv'
            }
}

IMG_PATH = {
    'MEE': {'train': 'data/images/',
            'dev': 'data/images/',
            'test': 'data/images/'}
}

AUX_PATH = {
    'MEE': {
                'train': 'data/object',
                'dev': 'data/object',
                'test': 'data/object'}
}

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Select', default='ME', type=str, help="EE, NER or ME.")
    parser.add_argument('--preprocessed', default=False, help="preprocess images.")
    parser.add_argument('--labeled', default=False, help="Select = 'NER', Whether or not to label")
    parser.add_argument('--labeled_data_path', default='data/val.json', type=str, help="labeled = True, label data_path")
    parser.add_argument('--caption_dataset', default='./data/Caption_MEE.csv', type=str, help="The name of image_caption")
    parser.add_argument('--role_path', default='./data/description.csv', type=str, help="The name of description")
    parser.add_argument('--dataset_name', default='MEE', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=50, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.06, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1234, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=4, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    # parser.add_argument('--load_path', default='ckpt/EE/best_model.pth', type=str, help="Load model from load_path")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default='ckpt/ME', type=str, help="save model at save_path")
    parser.add_argument('--write_path', default='ckpt/ME', type=str, help="do_test = True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', default=True)
    parser.add_argument('--do_train', default=True)
    parser.add_argument('--only_test', default=False)
    parser.add_argument('--max_seq', default=80, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resources.")


    args = parser.parse_args()

    data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.Select], TRAINER_CLASSES[args.Select]
    data_process, dataset_class = DATA_PROCESS[args.Select]

    def preprocess_and_save_images(img_path, aux_img_path, transform, save_path, save_aux_dir):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('预处理images开始！')
        for img_name in tqdm(os.listdir(img_path)):
            img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
            img = transform(img)
            torch.save(img, os.path.join(save_path, img_name + '.pt'))
        print('完成！')
        print("预处理aux_images开始！")
        if not os.path.exists(save_aux_dir):
            os.makedirs(save_aux_dir)
        for aux_img_name in tqdm(os.listdir(aux_img_path)):
            if aux_img_name.endswith('.png'):
                aux_img_file_path = os.path.join(aux_img_path, aux_img_name).replace('\\', '/')
                if os.path.isfile(aux_img_file_path):  # Check if aux_img_file_path is a file
                    aux_img = Image.open(aux_img_file_path).convert('RGB')
                    aux_img = transform(aux_img)
                    torch.save(aux_img, os.path.join(save_aux_dir, aux_img_name + '.pt'))
        print("完成！")

    # Define your transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if args.preprocessed:
        # Preprocess and save images
        preprocess_and_save_images('data/images', 'data/object', transform, 'data/preprocessed_images', 'data/preprocessed_images/aux_images')

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        # args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    logdir = "logs/" + args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir)
    writer=None

    if not args.use_prompt:
        img_path, aux_path = None, None

    caption_data = {}
    with open(args.caption_dataset, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row[next(iter(row))]
            caption_data[key] = row
    if args.labeled:
        data_path['test'] = args.labeled_data_path
    processor = data_process(data_path, args.bert_name, args.max_seq)
    train_dataset = dataset_class(args, caption_data, processor, transform, img_path, aux_path, args.max_seq, sample_ratio=args.sample_ratio, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    dev_dataset = dataset_class(args, caption_data, processor, transform, img_path, aux_path, args.max_seq, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    test_dataset = dataset_class(args, caption_data, processor, transform, img_path, aux_path, args.max_seq, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    if args.Select == 'EE':  # EE task
        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer
        model = EEModel(num_labels, tokenizer, args=args)
        model.to(args.device)
        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                          processor=processor, args=args, logger=logger, writer=writer)

    elif args.Select == 'NER':   # NER task
        NER_label_mapping = processor.get_label_mapping()
        NER_label_list = list(NER_label_mapping.keys())
        model = NERModel(NER_label_list, args)
        model.to(args.device)
        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                          label_map=NER_label_mapping, args=args, logger=logger, writer=writer)
    else:
        ROLE_label_mapping = processor.get_label_mapping()
        ROLE_label_list = list(ROLE_label_mapping.keys())
        model = MEModel(ROLE_label_list, args)
        model.to(args.device)
        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                          label_map=ROLE_label_mapping, args=args, logger=logger, writer=writer)

    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

