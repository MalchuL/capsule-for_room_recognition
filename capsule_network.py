"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import torchvision
import batches
import fnmatch
import PIL.Image
import torchvision.transforms as tr
import os

BATCH_SIZE = 100
NUM_CLASSES = 5
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3


def pred(path, dest):
    dirs = os.listdir(path)
    preprocess = tr.Compose([tr.Resize((299, 299)), tr.ToTensor()])
    f = open(dest, "w")
    for file in dirs:
        if fnmatch.fnmatch(file, '*.jpg'):
            img = PIL.Image.open(path + file)
            x = preprocess(img.convert('RGB'))
            x = torch.stack([x,x], 0).type(torch.FloatTensor).cuda()
            x.requires_grad = False
            res = eval(x)
            f.write(file + ' ' + define(res) + '\n')
    f.close()


def define(label):
    if (label == 0):
        return "Gostinnaya komnata"
    elif (label == 1):
        return "Kuhnya"
    elif (label == 2):
        return "Sanuzel"
    elif (label == 3):
        return "Spalnya"
    else:
        return "Ne raspoznano"

def eval(x):
    with torch.no_grad():
        predicted=model(x)
        return np.argmax(predicted, axis=1)

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        inception = torchvision.models.inception_v3(pretrained=True)
        inception.requires_grad = False
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=3, stride=1)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=512, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            #reconstruction size
            nn.Linear(1024, 3*299*299),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):

        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        x = F.selu(self.conv1(x), inplace=True)
        x = F.selu(self.conv2(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            print(classes.size())
            # In all batches, get the most active capsule.
            if len(classes.size())==2:
                classes= classes.unsqueeze(0)
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    #from tqdm import tqdm
    #import torchnet as tnt
    model = CapsuleNet()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model = model.cuda()


    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters(),lr=1e-5)

    capsule_loss = CapsuleLoss()


    dataset = batches.HackatonDataset('./datasets','.jpg')

    def processor(sample):
        data, labels, training = sample


        labels = torch.LongTensor(labels)

        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)
        print(labels,classes)
        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes

    def train(path_to_save='./model.cktp', batch_size=5):
        def get_batch_func(batch_size):
            data, output = dataset.get_train_batch(batch_size)
            return data,torch.from_numpy(np.array(output)).type(torch.LongTensor)

        def get_test_batch(iterations, batch_size):
            data, output = dataset.get_train_batch(batch_size)
            return data, torch.from_numpy(np.array(output)).type(torch.LongTensor)

        def save():
            torch.save(model.state_dict(), path_to_save)

        def resume():
            try:
                model.load_state_dict(torch.load(path_to_save))
            except Exception as ex:
                print('no saved model: ', ex)

        resume()
        for epoch in range(10000):

            if epoch > 0 and get_test_batch:
                save()
                print("start testing")
                loss = 0

                input, output = get_test_batch(iteration, 10)
                if True:
                    input, output = input.cuda(), output

                with torch.no_grad():
                    loss, classes = processor([input, output, True])

                    current_loss = loss.item()
                    print('test loss', current_loss)



            for iteration in range(100):

                input, output = get_batch_func(batch_size)
                if True:
                    input, output = input.cuda(), output

                loss, classes = processor([input,output,True])
                print(loss.item())

                loss.backward()
                optimizer.step()

                del loss

    import argparse
    parser = argparse.ArgumentParser(description='capsules')

    parser.add_argument('--use_eval', default='0', type=int,
                        help='0 - eval, 1 - train')

    FLAGS = parser.parse_args()

    if FLAGS.use_eval == 0:
        pred('./test/', 'result.txt')
    else:
        #pred('./test/', 'result.txt')
        train()







