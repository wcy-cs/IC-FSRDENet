import argparse
parser = argparse.ArgumentParser(description='FaceSR')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--cuda_name', type=str, default='0')
parser.add_argument('--gpu_ids', type=int, default=1)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--device', default='cuda')

parser.add_argument('--dir_data', type=str, default='/data/',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='train',
                    help='train dataset name')
parser.add_argument('--data_val', type=str, default='val',
                    help='val dataset name')
parser.add_argument('--scale', type=int, default=8,
                    help='super resolution scale')

parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--low_light', action='store_true',
                    help='use low light dataset')
parser.add_argument('--small', action='store_true',
                    help='use low light dataset')

# Model specifications
parser.add_argument('--model', default='MYNET',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--n_resblocks', type=int, default=6,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.2,
                    help='residual scaling')
parser.add_argument('--large', action="store_true",
                    help='the input is as large as output or not')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')

parser.add_argument('--test_only', action='store_true',# default=True,
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--n_steps', type=int, default=30,
                    help='学习率衰减倍数')

# Log specifications
parser.add_argument('--root', type=str, default='')
parser.add_argument('--save_path', type=str, default='./experiment',
                    help='file path to save model')
parser.add_argument("--writer_name", type=str, default="mynet",
                    help="the name of the writer")

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
# args.scale = list(map(lambda x: int(x), args.scale.split('+')))


if args.epochs == 0:
    args.epochs = 40

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

