import os
import shutil
import argparse

command_list = ['zero', 'one', 'two', 'three', 'four', 'five']

def move_files(original_fold, data_fold, data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            if vals[0] not in command_list:
                continue
            dest_fold = os.path.join(data_fold, vals[0])
            if not os.path.exists(dest_fold):
                os.mkdir(dest_fold)
            shutil.move(os.path.join(original_fold, line[:-1]), os.path.join(data_fold, line[:-1]))


def create_train_fold(original_fold, data_fold, test_fold):
    dir_names = list()
    for file in os.listdir(test_fold):
        if os.path.isdir(os.path.join(test_fold, file)):
            dir_names.append(file)

    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(test_fold, file)) and file in dir_names:
            shutil.move(os.path.join(original_fold, file), os.path.join(data_fold, file))


def make_dataset(commands_fold, out_path):
    test_path = os.path.join(commands_fold, 'testing_list.txt')

    test_fold = os.path.join(out_path, 'test')
    train_fold = os.path.join(out_path, 'train')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(test_fold):
        os.mkdir(test_fold)
    if not os.path.exists(train_fold):
        os.mkdir(train_fold)

    move_files(commands_fold, test_fold, test_path)
    create_train_fold(commands_fold, train_fold, test_fold)


parser = argparse.ArgumentParser(description='Make commands dataset.')
parser.add_argument('--commands_fold', help='the path to the root folder of the commands dataset.')
parser.add_argument('--out_path', default='dataset', help='the path where to save the files splitted to folders.')

if __name__ == '__main__':
    args = parser.parse_args()
    make_dataset(args.commands_fold, args.out_path)
