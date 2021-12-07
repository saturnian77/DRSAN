from argparse import ArgumentParser
import os
import build

parser = ArgumentParser()

parser.add_argument('--gpu', type=str, dest='gpu', default='0', help='Choose GPU to use')
parser.add_argument('--model', type=str, dest='model', default='32s', help='Choose model: 32s, 32m, 48s, 48m')
parser.add_argument('--scale', type=str, dest='scale', default='x2', help='Choose scale: x2, x3, x4')
parser.add_argument('--dataset', type=str, dest='dataset', default='Set5', help='Dataset: Set5, Set14, B100, Urban100')

args = parser.parse_args()



os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    ckpt_path = './ckpt/' + args.scale + '/' + 'DRSAN_' + args.model + '/'
    model = build.Build(ckpt_path, args.model, args.scale, args.dataset)
    model.test()

if __name__ == '__main__':
    main()