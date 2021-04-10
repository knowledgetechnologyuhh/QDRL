import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from vr.models import maced_net, filmed_net
import argparse
from vr.models import MAC, FiLMedNet, FiLMGen

import sys
from os.path import dirname, join, abspath
import argparse
import shutil

import math
from copy import deepcopy

import pytorch_lightning as pl
import torch
from munch import Munch
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import torchvision

from qsr_learning.data import DRLDataset
# from qsr_learning.models import DRLNet

from entities_ralations import entity_names, excluded_entity_names_18, excluded_entity_names_1left, \
relation_names_abs_intri, excluded_relation_names_all4_abs_intri, excluded_relation_names_above_below_abs_intri, \
relation_names_rel, excluded_relation_names_all4_rel, excluded_relation_names_front_behind_rel

from config import mac_language_encoder_kwargs, film_language_encoder_kwargs, mac_kwargs, film_kwargs, train_kwargs, output_kwargs

# relation_names, excluded_relation_names_all4, excluded_relation_names_front_behind

from torch.utils.tensorboard import SummaryWriter

import datetime
import dateutil
import dateutil.tz

import pdb

parser = argparse.ArgumentParser()

from qsr_learning.entity import emoji_names

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def invert_dict(d):
    return {v: k for k, v in d.items()}

parser.add_argument('--reference_type', default='intrinsic', choices=["intrinsic", "absolute", "relative"])
parser.add_argument('--num_entity', default=2, choices=[2, 3, 5], type=int)
parser.add_argument('--excluded_entity', default='18', choices=["18", "1left"])
parser.add_argument('--excluded_ralation', default='all4', choices=["all4", "front_behind"])
parser.add_argument('--model_type', default='film', choices=["film", "mac"])
parser.add_argument('--num_modules', default=4, required=True, type=int)
parser.add_argument('--torch_random_seed', default=0, required=True, type=int)
parser.add_argument('--image_size', default=224, required=True, type=int)
parser.add_argument('--emoji_size', default=24, required=True, type=int)
parser.add_argument('--total_epochs', default=10, required=True, type=int)
parser.add_argument('--train_based_on_checkpoint', default=0, required=True, type=int)
parser.add_argument('--test', default=0, type=int)
parser.add_argument('--test_mode', default='comp', choices=["comp", "std"])
parser.add_argument('--checkpoint_dirname', default=None)
parser.add_argument('--checkpoint_filename', default=None)


"""
def get_state(m):
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state
"""
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def check_accuracy(language_encoder, model, model_type, validation_loader, num_val_samples=None):
    language_encoder.eval()
    model.eval()
    num_correct, num_samples = 0, 0
    for batch in validation_loader:
        (images, questions, answers) = batch # images.shape: [32, 3, 224, 224]
        questions_var = Variable(questions.to(device))
        questions_feat = language_encoder(questions_var)
        images_var = Variable(images.to(device))
        answers_var = Variable(answers.type(torch.LongTensor).to(device))
        if model_type == "mac":
            scores = model(images_var, questions_feat, isTest=True)
        elif model_type == "film":
            scores = model(images_var, questions_feat)

        if scores is not None:
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == answers).sum()
            num_samples += preds.size(0)
        
        if num_val_samples is not None and num_samples >= num_val_samples:
            break

    language_encoder.train()
    model.train()
    acc = float(num_correct) / num_samples
    return acc

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def set_datadir(root_datadir, model_name, num_modules):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    datadir = os.path.join(
        root_datadir,
        "{}_{}_num_modules_{}".format(now, model_name, num_modules),
    )
    mkdir_p(datadir)
    print("Saving output to: {}".format(datadir))
    return datadir

def load_checkpoint(checkpoint):
    if os.path.isfile(checkpoint):
        print("=> loading checkpoint '{}'".format(checkpoint))
        cp = torch.load(checkpoint)
        return cp
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint))
        raise Exception("No checkpoint found!")


def test(args):

    suffix0 = "torch_random_seed_{}".format(args.torch_random_seed)
    suffix1 = "num_entity_{}".format(args.num_entity) # "without_coords"
    suffix2 = "excluded_entity_{}_excluded_relation_{}".format(args.excluded_entity, args.excluded_ralation)
    suffix3 = "image_size_{}_emoji_size_{}".format(args.image_size, args.emoji_size)
    output_kwargs['root_datadir'] = join(dirname(abspath(__file__)), os.pardir, "data", args.reference_type, suffix0, suffix1, suffix2, suffix3)

    # log file
    """
    test_logfile = os.path.join(log_dir,"test_{}.log".format(args.test_mode))
    sys.stdout = Logger(logfile=test_logfile)
    print("args: {}".format(args))
    """

    if args.reference_type == "intrinsic" or args.reference_type == "absolute":
        relation_names = relation_names_abs_intri
        excluded_relation_names_all4 = excluded_relation_names_all4_abs_intri
        excluded_relation_names_front_behind = excluded_relation_names_above_below_abs_intri
        pass
    elif args.reference_type == "relative":
        relation_names = relation_names_rel
        excluded_relation_names_all4 = excluded_relation_names_all4_rel
        excluded_relation_names_front_behind = excluded_relation_names_front_behind_rel
        pass
    else:
        raise

    if args.excluded_entity == "18":
        excluded_entity_names = excluded_entity_names_18
        pass
    elif args.excluded_entity == "1left":
        excluded_entity_names = excluded_entity_names_1left
        pass
    else:
        raise

    if args.excluded_ralation == "all4":
        excluded_relation_names = excluded_relation_names_all4
    elif args.excluded_ralation == "front_behind":
        excluded_relation_names = excluded_relation_names_front_behind
        pass
    else:
        raise

    # model part
    checkpoint_path = os.path.join(output_kwargs['root_datadir'], args.checkpoint_dirname, 'Checkpoints', 'ckpt_{}.pth.tar'.format(args.checkpoint_filename))

    language_encoder = None
    model = None
    vocab = None

    cp = load_checkpoint(checkpoint_path)
    vocab = cp['vocab']

    model_type = args.model_type
    if model_type == "mac":
        mac_language_encoder_kwargs['null_token'] = vocab['question_token_to_idx']['NULL']
        mac_language_encoder_kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
        mac_language_encoder_kwargs['num_modules'] = args.num_modules
        mac_kwargs['vocab'] = vocab
        mac_kwargs['feature_dim'] = [3,args.image_size,args.image_size]
        mac_kwargs['num_modules'] = args.num_modules
        language_encoder = FiLMGen(**mac_language_encoder_kwargs).to(device)
        model = MAC(**mac_kwargs).to(device)
    elif model_type == "film":
        film_language_encoder_kwargs['null_token'] = vocab['question_token_to_idx']['NULL']
        film_language_encoder_kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
        film_language_encoder_kwargs['num_modules'] = args.num_modules
        film_kwargs['vocab'] = vocab
        film_kwargs['feature_dim'] = [3,args.image_size,args.image_size]
        film_kwargs['num_modules'] = args.num_modules
        film_kwargs['condition_pattern'] = [1] * args.num_modules
        language_encoder = FiLMGen(**film_language_encoder_kwargs).to(device)
        model = FiLMedNet(**film_kwargs).to(device)
    
    language_encoder.load_state_dict(cp["language_encoder_state"])
    model.load_state_dict(cp["model_state"])

    # dataset related
    cfg = Munch()
    cfg.dataset = Munch(
        vocab=list(vocab["question_token_to_idx"].keys()),
        entity_names=entity_names,
        relation_names=relation_names,
        num_entities=args.num_entity,
        frame_of_reference= args.reference_type, # "intrinsic" or "absolute"
        w_range=(args.emoji_size, args.emoji_size), #(8, 8) (16, 16) (24, 24) (32, 32)
        h_range=(args.emoji_size, args.emoji_size), #(8, 8)
        theta_range=(0, 2 * math.pi),
        add_bbox=False,
        add_front=False,
        transform=None,
        canvas_size=(args.image_size, args.image_size), #(224, 224) (128, 128) (64, 64)
        num_samples=10 ** 6,
        root_seed=0,
    )

    if args.test_mode == "std":
        test_dataset = DRLDataset(
            **{
                **cfg.dataset,
                **dict(
                    excluded_entity_names=excluded_entity_names,
                    excluded_relation_names=excluded_relation_names,
                    num_samples=10 ** 4,
                    root_seed=10 ** 7,
                ),
            }
        )
    elif args.test_mode == "comp":
        test_dataset = DRLDataset(
            **{
                **cfg.dataset,
                **dict(
                    entity_names=excluded_entity_names,
                    excluded_entity_names=[],
                    relation_names=excluded_relation_names,
                    excluded_relation_names=[],
                    num_samples=10 ** 4,
                    root_seed=10 ** 7,
                ),
            }
        )
    else:
        raise

    cfg.data_loader = Munch(
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset, **{**cfg.data_loader, "shuffle": False}
    )

    test_acc = check_accuracy(language_encoder, model, model_type, test_loader, num_val_samples=None)
    print("test_acc: {}".format(test_acc))
    pass


def train(args):
    cfg = Munch()
    if args.reference_type == "intrinsic" or args.reference_type == "absolute":
        relation_names = relation_names_abs_intri
        excluded_relation_names_all4 = excluded_relation_names_all4_abs_intri
        excluded_relation_names_front_behind = excluded_relation_names_above_below_abs_intri
        pass
    elif args.reference_type == "relative":
        relation_names = relation_names_rel
        excluded_relation_names_all4 = excluded_relation_names_all4_rel
        excluded_relation_names_front_behind = excluded_relation_names_front_behind_rel
        pass
    else:
        raise

    if args.excluded_entity == "18":
        excluded_entity_names = excluded_entity_names_18
        pass
    elif args.excluded_entity == "1left":
        excluded_entity_names = excluded_entity_names_1left
        pass
    else:
        raise

    if args.excluded_ralation == "all4":
        excluded_relation_names = excluded_relation_names_all4
    elif args.excluded_ralation == "front_behind":
        excluded_relation_names = excluded_relation_names_front_behind
        pass
    else:
        raise

    null_token = ["NULL"]
    question_tokens = entity_names + relation_names + null_token + ["as_seen_from"]
    word2idx = {}
    idx2word = {}
    for idx, word in enumerate(sorted(question_tokens)):
        word2idx[word] = idx
        idx2word[idx] = word
    vocab = {
        "question_token_to_idx": word2idx, 
        'question_idx_to_token': idx2word,
        "answer_token_to_idx": {
            "true": 1,
            "false": 0
        },
    }
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    stats = {
        'train_losses': [], 'train_accs': [], 'val_std_accs': [], 'val_comp_accs': [], 'model_iter':0, 'model_epoch': 0,
    }

    cfg.dataset = Munch(
        vocab=question_tokens,
        entity_names=entity_names,
        relation_names=relation_names,
        num_entities=args.num_entity,
        frame_of_reference= args.reference_type, # "intrinsic" or "absolute"
        w_range=(args.emoji_size, args.emoji_size), #(8, 8) (16, 16) (24, 24) (32, 32)
        h_range=(args.emoji_size, args.emoji_size), #(8, 8)
        theta_range=(0, 2 * math.pi),
        add_bbox=False,
        add_front=False,
        transform=None,
        canvas_size=(args.image_size, args.image_size), #(224, 224) (128, 128) (64, 64)
        num_samples=10 ** 6,
        root_seed=0,
    )

    train_dataset = DRLDataset(
        **{
            **cfg.dataset,
            **dict(
                excluded_entity_names=excluded_entity_names,
                excluded_relation_names=excluded_relation_names,
                num_samples=10 ** 6,
                root_seed=0,
            ),
        }
    )
    validation_dataset_standard = DRLDataset(
        **{
            **cfg.dataset,
            **dict(
                excluded_entity_names=excluded_entity_names,
                excluded_relation_names=excluded_relation_names,
                num_samples=10 ** 4,
                root_seed=train_dataset.num_samples,
            ),
        }
    )
    validation_dataset_compositional = DRLDataset(
        **{
            **cfg.dataset,
            **dict(
                entity_names=excluded_entity_names,
                excluded_entity_names=[],
                relation_names=excluded_relation_names,
                excluded_relation_names=[],
                num_samples=10 ** 4,
                root_seed=train_dataset.num_samples
                + validation_dataset_standard.num_samples,
            ),
        }
    )

    cfg.data_loader = Munch(
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    train_loader = DataLoader(train_dataset, **cfg.data_loader)
    validation_loader_standard = DataLoader(
        validation_dataset_standard, **{**cfg.data_loader, "shuffle": False}
    )
    validation_loader_compositional = DataLoader(
        validation_dataset_compositional, **{**cfg.data_loader, "shuffle": False}
    )
    
    suffix0 = "torch_random_seed_{}".format(args.torch_random_seed)
    suffix1 = "num_entity_{}".format(args.num_entity) # "without_coords"
    suffix2 = "excluded_entity_{}_excluded_relation_{}".format(args.excluded_entity, args.excluded_ralation)
    suffix3 = "image_size_{}_emoji_size_{}".format(args.image_size, args.emoji_size)
    
    output_kwargs['root_datadir'] = join(dirname(abspath(__file__)), os.pardir, "data", args.reference_type, suffix0, suffix1, suffix2, suffix3)

    # set dirs
    data_dir = set_datadir(output_kwargs['root_datadir'], model_name=args.model_type, num_modules=args.num_modules)
    log_dir = os.path.join(data_dir, "Log")
    mkdir_p(log_dir)
    ckp_dir = os.path.join(data_dir, "Checkpoints")
    mkdir_p(ckp_dir)
    code_dir = os.path.join(data_dir, "Code")
    mkdir_p(code_dir)
    # copy code
    for filename in os.listdir(join(dirname(abspath(__file__)))):
        if filename.endswith(".py"):
            shutil.copy(join(dirname(abspath(__file__)), filename), code_dir)
    # tensorboard log
    writer = SummaryWriter(log_dir=log_dir)
    # log file
    train_logfile = os.path.join(log_dir,"train.log")
    sys.stdout = Logger(logfile=train_logfile)

    """
    with open(os.path.join(log_dir, 'args.txt'), 'w') as dst:
        print(args, file=dst)
    """

    print("Saving output to: {}".format(data_dir))

    print("len(entity_names): {}".format(len(entity_names)))
    print("len(excluded_entity_names): {}".format(len(excluded_entity_names)))
    print("relation_names: {}".format(relation_names))
    print("excluded_relation_names: {}".format(excluded_relation_names))

    print("vocab[\'question_token_to_idx\']: {}".format(vocab["question_token_to_idx"]))
    print("args: {}".format(args))

    model_type = args.model_type
    if model_type == "mac":
        mac_language_encoder_kwargs['null_token'] = vocab['question_token_to_idx']['NULL']
        mac_language_encoder_kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
        mac_language_encoder_kwargs['num_modules'] = args.num_modules
        mac_kwargs['vocab'] = vocab
        mac_kwargs['feature_dim'] = [3,args.image_size,args.image_size]
        mac_kwargs['num_modules'] = args.num_modules

        language_encoder = FiLMGen(**mac_language_encoder_kwargs).to(device)
        model = MAC(**mac_kwargs).to(device)
    elif model_type == "film":
        film_language_encoder_kwargs['null_token'] = vocab['question_token_to_idx']['NULL']
        film_language_encoder_kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
        film_language_encoder_kwargs['num_modules'] = args.num_modules
        film_kwargs['vocab'] = vocab
        film_kwargs['feature_dim'] = [3,args.image_size,args.image_size]
        film_kwargs['num_modules'] = args.num_modules
        film_kwargs['condition_pattern'] = [1] * args.num_modules

        language_encoder = FiLMGen(**film_language_encoder_kwargs).to(device)
        model = FiLMedNet(**film_kwargs).to(device)

    language_encoder_optimizer = torch.optim.Adam(language_encoder.parameters(), lr=0.0001)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    iteration = 0
    if args.train_based_on_checkpoint == 1:
        checkpoint_path = os.path.join(output_kwargs['root_datadir'], args.checkpoint_dirname, 'Checkpoints', 'ckpt_{}.pth.tar'.format(args.checkpoint_filename))
        cp = load_checkpoint(checkpoint_path)
        language_encoder.load_state_dict(cp["language_encoder_state"])
        model.load_state_dict(cp["model_state"])
        language_encoder_optimizer.load_state_dict(cp["language_encoder_optimizer_state"])
        model_optimizer.load_state_dict(cp["model_optimizer_state"])
        iteration = cp['model_iter']

    # set train
    language_encoder.train()
    model.train()

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Image encoder
    """
    resnet = getattr(torchvision.models, 'resnet18')(pretrained=True)
    image_encoder = torch.nn.Sequential(*deepcopy(list(resnet.children())[:-3]))
    del resnet
    # Freeze the image encoder weights
    for param in image_encoder.parameters():
        param.requires_grad = False
    image_encoder.to(device).eval()
    """

    train_kwargs["total_epochs"] = args.total_epochs

    epoch = 0
    running_loss = 0.0
    
    while epoch < train_kwargs["total_epochs"]:
        epoch += 1
        for batch in train_loader:
            iteration += 1
            (images, questions, answers) = batch # images.shape: [32, 3, 224, 224]
            questions_var = Variable(questions.to(device))
            questions_feat = language_encoder(questions_var)
            images_var = Variable(images.to(device))
            # images_feat = image_encoder(images_var) # images_feat.shape: [32, 256, 14, 14]
            answers_var = Variable(answers.type(torch.LongTensor).to(device))
            scores = model(images_var, questions_feat)
            loss = loss_fn(scores, answers_var)

            language_encoder_optimizer.zero_grad()
            model_optimizer.zero_grad()

            loss.backward()
            language_encoder_optimizer.step()
            model_optimizer.step()

            if iteration % output_kwargs['record_loss_every_n_iter'] == 0:
                running_loss += loss.item()
                avg_loss = running_loss / output_kwargs['record_loss_every_n_iter']
                stats['train_losses'].append(avg_loss)
                writer.add_scalar("avg_loss", avg_loss, iteration)
                running_loss = 0.0
            else:
                running_loss += loss.item()

            if iteration % output_kwargs['evaluation_every_n_iter'] == 0:
                num_val_samples = train_kwargs["num_val_samples"]
                train_acc = check_accuracy(language_encoder, model, model_type, train_loader, num_val_samples=num_val_samples)
                val_std_acc = check_accuracy(language_encoder, model, model_type, validation_loader_standard, num_val_samples=num_val_samples)
                val_comp_acc = check_accuracy(language_encoder, model, model_type, validation_loader_compositional, num_val_samples=num_val_samples)
                stats['train_accs'].append(train_acc)
                stats['val_std_accs'].append(val_std_acc)
                stats['val_comp_accs'].append(val_comp_acc)
                writer.add_scalar("train_accs", train_acc, iteration)
                writer.add_scalar("val_std_accs", val_std_acc, iteration)
                writer.add_scalar("val_comp_accs", val_comp_acc, iteration)
                stats['model_iter'] = iteration
                stats['model_epoch'] = epoch
                print("iteration: {}   train_acc: {}   val_std_accs: {}  val_comp_accs: {}".format(iteration, train_acc, val_std_acc, val_comp_acc))

            if iteration % output_kwargs['checkpoint_every_n_iter'] == 0:
                language_encoder_state = language_encoder.state_dict()
                language_encoder_optimizer_state = language_encoder_optimizer.state_dict()
                model_state = model.state_dict()
                model_optimizer_state = model_optimizer.state_dict()
                checkpoint = {
                    'language_encoder_state': language_encoder_state,
                    'language_encoder_optimizer_state': language_encoder_optimizer_state,
                    'model_state': model_state,
                    'model_optimizer_state': model_optimizer_state,
                    'vocab': vocab
                }
                for k, v in stats.items():
                    checkpoint[k] = v

                print('Saving checkpoint to %s' % ckp_dir)
                torch.save(checkpoint, "{}/ckpt_{}.pth.tar".format(ckp_dir, iteration))
                # torch.save(checkpoint, "{}/ckpt_final.pth.tar".format(ckp_dir))

                pass

    # save the final checkpoint
    checkpoint = {
        'language_encoder_state': language_encoder_state,
        'language_encoder_optimizer_state': language_encoder_optimizer_state,
        'model_state': model_state,
        'model_optimizer_state': model_optimizer_state,
        'vocab': vocab
    }
    for k, v in stats.items():
        checkpoint[k] = v
    print('Saving checkpoint to %s' % ckp_dir)
    torch.save(checkpoint, "{}/ckpt_final.pth.tar".format(ckp_dir))


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.torch_random_seed)
    if args.test == 1:
        test(args)
    else:
        train(args)

pass








