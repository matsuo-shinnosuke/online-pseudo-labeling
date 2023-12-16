import argparse
from pathlib import Path


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--model_backbone', default='resnet18', type=str)
    parser.add_argument('--is_pretrain', default=0, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--mini_batch', default=4, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--num_epochs', default=400, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--is_early_stopping', default=0, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--is_train', default=1, type=int)
    parser.add_argument('--is_eval', default=1, type=int)

    parser.add_argument('--dataset_dir', default='data/', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--val_size', default=0.2, type=float)
    parser.add_argument('--output_dir', default='result/', type=str)

    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_instances', default=1024, type=int)
    parser.add_argument('--num_bags', default=100, type=int)

    parser.add_argument('--is_pertur', default=1, type=int)
    parser.add_argument('--is_cumul', default=1, type=int)
    parser.add_argument('--sigma', default=1, type=int)
    parser.add_argument('--eta', default=5, type=int)
 
    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset_dir)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args
