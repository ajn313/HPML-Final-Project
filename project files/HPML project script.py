import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import numpy as np
from time import perf_counter_ns
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context 


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--nworkers',default=2,type=int,help='Number of data loader workers')
parser.add_argument('--gpus',default=1,type=int,help='Set to zero for single GPU')
parser.add_argument('--batchsize',default=2048,type=int,help='Set batch size')
parser.add_argument('--lr',default=0.0001,type=float,help='Set learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 1  # start from epoch 1
best_acc = 0  # best test accuracy

torch.cuda.empty_cache()

img_size = 64
n_epochs = 50
batch_size = args.batchsize
learning_rate = args.lr*(batch_size/64)
nworkers = args.nworkers

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

term_width = 80
TOTAL_BAR_LENGTH = 65.
last_time = time.time()

def progress_bar(current, total, msg=None):
    global last_time
    begin_time = last_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    last_time = time.time()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
#set batch size to what was passed in argument, was originally 128
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=nworkers)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=nworkers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')

net = ResNet18()
if device == 'cuda':
    net = torch.nn.DataParallel(net) #enable data parallelism for multiple GPUs
    cudnn.benchmark = True
net = net.to(device)
#Checkpoint resume goes here



crossen = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

bil = 1000000000.0

top_train_acc = 0 
train_times=[]
train_losses=[]
train_accs=[]
load_times=[]
compute_times=[]
top_train_loss = 100


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global top_train_acc #code to record top 1 training accuracy
    global train_times #code to record average train time
    global train_losses #code to record average train loss
    global top_train_loss #code to record top 1 train loss
    global load_times #code to record data loader times
    top_epoch_acc = 0
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    #count = 0
    datatimesum = 0.0
    traintimesum = 0.0
    dataloadstart=perf_counter_ns() #data loading timer start for C2
    computetotal=0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        dataloadstop=perf_counter_ns()  #data loading timer stop for C2
        datatimesum += (dataloadstop-dataloadstart)/bil
        traintimestart = perf_counter_ns() #start training timer
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = crossen(outputs, targets)
        loss.backward()
        optimizer.step()
        traintimestop =perf_counter_ns() #stop training timer
        computestart = perf_counter_ns()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        curr_acc = 100.*correct/total
        traintimesum += (traintimestop-traintimestart)/bil
        if (curr_acc>top_train_acc): #code to record top 1 training accuracy
            top_train_acc = curr_acc
        train_times.append(traintimesum) #code to record average train time
        if (curr_acc>top_epoch_acc): #code to record top 1 training accuracy for epoch
            top_epoch_acc = curr_acc
        computestop = perf_counter_ns()
        computetotal += (computestop-computestart)/bil
        #if (count%20 == 0):  #only print progress bar every 20 steps, reduce clutter on Spyder
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #count += 1
        dataloadstart=perf_counter_ns() #data loading timer start for C2
    train_losses.append(train_loss/(batch_idx+1)) #code to record average train loss
    train_accs.append(top_epoch_acc)
    if ((train_loss/(batch_idx+1))<top_train_loss): #code to record top train loss
        top_train_loss = train_loss/(batch_idx+1)
    load_times.append(datatimesum) #code to record data loader times
    compute_times.append(computetotal)
    print("\nEpoch data loader time:  ", datatimesum," sec")
    print("Epoch Compute Time: ",computetotal," sec")
    print("Epoch training time: ",traintimesum," sec")
    print("Epoch total time: ",datatimesum+computetotal+traintimesum," sec")
    print("Epoch Average Training Time: ",traintimesum/(batch_idx+1)," sec")
    print("Epoch Average Loss: ",train_loss/(batch_idx+1))
    print("Top Epoch Accuracy: ",top_epoch_acc,"%")

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = crossen(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print("Val Acc: ",acc,"%\n")
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if (args.gpus == 1):
            torch.save(state, './checkpoint/ckpt.pth')
        else:
            torch.save(state, './checkpoint/ckpt1gpu.pth')
        best_acc = acc

runtimes = []

if start_epoch <n_epochs-1:
    for epoch in range(start_epoch, n_epochs+1):
        totaltimestart = perf_counter_ns() #total time counter for C2
        train(epoch)
        totaltimestop= perf_counter_ns() #total time counter for C2
        timestep = (totaltimestop-totaltimestart)/bil
        runtimes.append(timestep)
        print("Epoch ",epoch," Total Runtime: ",timestep," sec\n")
        print("Testing:\n")
        test(epoch)
        scheduler.step()

    print("\nTotal Data Load Time: ",np.sum(load_times)," sec")
    print("Total Compute Time: ",np.sum(compute_times),"sec")
    print("Average Compute Time: ",np.mean(compute_times),"sec")
    print("Total Runtime of Training Epochs: ",np.sum(runtimes)," sec")
    print("Average Runtime of Training Epochs: ",np.mean(runtimes)," sec")
    print("Top Training Accuracy: ",top_train_acc,"%")
    print("Average Train Time: ",np.mean(train_times)," sec")
    print("Average Train Loss: ",np.mean(train_losses))
    print("Top train loss: ",top_train_loss,"\n")
else:
    print("Optimal Targeter Loaded!\n")
    
print("Building GAN...\n")
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(100, 512*7*7, bias = False),
            nn.BatchNorm1d(512*7*7),
            nn.LeakyReLU(0.3),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        y = self.layer1(x)
        y = y.view((-1,512,7,7))
        y = self.layer2(y)
        return y

class Discr(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(Discr, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(0.3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.dropout(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        out = self.sigmoid(out)
        return out


def Discriminator():
    return Discr(BasicBlock, [2, 2, 2, 2])

D = Discriminator()
G = Generator()
if device == 'cuda':
    D = torch.nn.DataParallel(D) #enable data parallelism for multiple GPUs
    G = torch.nn.DataParallel(G)
    cudnn.benchmark = True
D = D.to(device)
G = G.to(device)
G.apply(weights_init)
D.apply(weights_init)


criterion = nn.BCELoss().to(device)
D_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))


def generate_samples(G,latent_v,epoch):
  with torch.no_grad():
    samples = G(latent_v)
  samples = samples.cpu()
  images = torchvision.utils.make_grid(samples,normalize=True)
  if args.gpus == 1:
      name = "epoch " + str(epoch) + " lr "+ str(learning_rate) + ".png"
  else:
      name = "epoch " + str(epoch) + " 1 gpu.png"
  plt.imsave(name,images.numpy().transpose((1,2,0)))


D_loss_hist = []
G_loss_hist = []
gan_load_times=[]
gan_compute_times=[]
D_train_times=[]
G_train_times=[]
datatimesum=0.0
Dtraintimesum = 0.0
Gtraintimesum = 0.0
noise = lambda x: torch.randn((x,100), device = device)
gantimestart = perf_counter_ns()
for epoch in range(n_epochs):

  D_losses = []
  G_losses = []
  dataloadstart=perf_counter_ns()
  for batch, (images,labels) in enumerate(trainloader):
    dataloadstop=perf_counter_ns()
    datatimesum += (dataloadstop-dataloadstart)/bil
    real_samples = images.to(device)
    bsz = real_samples.size(0)
    target_ones = torch.ones((bsz, 1), device=device)
    target_zeros = torch.zeros((bsz, 1), device=device)
    target_nines = []
    for i in range(bsz):
      target_nines.append(9)
    target_nines = torch.Tensor(target_nines).to(device)
    latent_v = noise(bsz)
    
    Dtraintimestart = perf_counter_ns() #start training timer
    #Train Discriminator
    D.zero_grad()
    pred_real = D(real_samples)
    loss_real = criterion(pred_real,target_ones)
    with torch.no_grad():
      fake_samples = G(latent_v)
    pred_fake = D(fake_samples)
    loss_fake = criterion(pred_fake,target_zeros)
    if epoch < n_epochs/4:
      loss = (loss_real+loss_fake)/4
    else:
      loss = (loss_real+loss_fake)/2
    loss.backward()
    D_opt.step()
    Dtraintimestop = perf_counter_ns() #stop training timer
    Dtraintime = (Dtraintimestop-Dtraintimestart)/bil
    D_train_times.append(Dtraintime)
    Dtraintimesum+=(Dtraintime)
    D_losses.append(loss)
    
    Gtraintimestart = perf_counter_ns()
    #Train Generator
    G.zero_grad()
    generated = G(latent_v)
    classes = D(generated)
    gen_loss1 = criterion(classes, target_ones)
    targets = net(generated)
    gen_loss9 = crossen(targets, target_nines.long())
    gen_loss = gen_loss1*0.7+gen_loss9*0.3
    gen_loss.backward()
    G_opt.step()
    Gtraintimestop = perf_counter_ns() #stop training timer
    Gtraintime = (Gtraintimestop-Gtraintimestart)/bil
    G_train_times.append(Gtraintime)
    Gtraintimesum+=(Gtraintime)
    G_losses.append(gen_loss)
    dataloadstart=perf_counter_ns()
    
    
  print('Epoch {} Discriminator Loss: {:.3f}, Generator Loss: {:.3f}'.format((epoch + 1),
                                                           torch.mean(torch.FloatTensor(D_losses)),
                                                           torch.mean(torch.FloatTensor(G_losses))))
  print("D train time: ",Dtraintime)
  print("G train time: ",Gtraintime)
  print('Saving..')
  Dstate = {
      'D': D.state_dict(),
      'epoch': epoch,
  }
  Gstate = {
      'G': G.state_dict(),
      'epoch': epoch,
  }
  if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
  if (args.gpus == 1):
      torch.save(Dstate, './checkpoint/D.pth')
  else:
      torch.save(Dstate, './checkpoint/D1gpu.pth')
  if (args.gpus == 1):
      torch.save(Gstate, './checkpoint/G.pth')
  else:
      torch.save(Gstate, './checkpoint/G1gpu.pth')
  if(epoch == 9 or epoch == 29 or epoch == 49):
    latent_test = noise(5)
    generate_samples(G,latent_test, epoch+1)
  D_loss_hist.append(torch.mean(torch.FloatTensor(D_losses)))
  G_loss_hist.append(torch.mean(torch.FloatTensor(G_losses)))

gantimestop = perf_counter_ns()
gantimesum = (gantimestop-gantimestart)/bil

print("Total GAN runtime: ",gantimesum)
print("Total GAN data loader time: ", datatimesum)
print("Total Discriminator train time: ",Dtraintimesum)
print("Total Generator train time: ",Gtraintimesum)
print("Average Discriminator train time per epoch: ",np.mean(D_train_times))
print("Average Generator train time per epoch: ",np.mean(G_train_times))

print("D Loss hist:")
print(D_loss_hist)
print("\nG loss hist:")
print(G_loss_hist)
for i in range(5):
  latent_test = noise(5)
  with torch.no_grad():
      samples = G(latent_test)
      labels = net(samples)
  sample = samples.cpu()
  print(torch.argmax(labels,dim=1))
  images = torchvision.utils.make_grid(sample,normalize=True)
  if args.gpus == 1:
      name = "output lr " + str(learning_rate) + ".png"
      plt.imsave(name,images.numpy().transpose((1,2,0)))
  else:
      plt.imsave("output 1 gpu.png",images.numpy().transpose((1,2,0)))