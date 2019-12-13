import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import dataset
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import pprint



def cifar_trainer(args, tb):
    print(f"torch version: {torch.__version__}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    print(f"processing on {device} with {torch.cuda.device_count()} processing units")

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[125.3241, 122.9694, 113.8878], std=[62.9932, 62.0887, 66.7049]),
        ])

    init_train_data = datasets.CIFAR10('data', train=True, download=True, transform = transform)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

    class_label_mapping = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'

    ]

    print(f"initial train_data_size: {len(init_train_data)}")
    print(f"test_data_size: {len(test_data)}")

    num_of_init_training_samples = len(init_train_data)
    validation_size = 0.1
    training_size = 1 - validation_size
    splits = [int(training_size*num_of_init_training_samples), int(validation_size*num_of_init_training_samples)]
    print(f"split sizes of train and val: {splits}")

    train_data, val_data = dataset.random_split(init_train_data, splits)
    print(f"train_data_size : {len(train_data)}")
    print(f"val_data_size: {len(val_data)}")


    #get the model
    model = torch.hub.load('pytorch/vision:v0.4.2', 'densenet121', pretrained=True)
    model.classifier = nn.Linear(1024,10,bias=True)
    model = nn.DataParallel(model).to(device)
    # print(model)




    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size,shuffle=True,num_workers = 4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.validation_batch_size, shuffle = True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.test_batch_size, shuffle = False, num_workers = 1)
    # for item in train_loader:
    #     print(item)

    # random_item = next(iter(train_loader))
    # print(random_item[0].shape)
    # print(len(random_item[1]))

    # random_img = random_item[0][0]
    # random_img = random_img.numpy()
    # print(random_img.shape)
    # random_img = np.transpose(random_img, (1,2,0))
    # random_img_label = random_item[1][0]

    # print(random_img.shape)
    # print(class_label_mapping[random_img_label])

    # plt.imshow(random_img)
    # plt.show()

    loss_criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov = True, weight_decay=args.weight_decay)

    num_of_epochs = args.epochs
    milestones = [0.5*num_of_epochs, 0.75*num_of_epochs]
    milestones = np.array(milestones, dtype=np.uint8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer = optimizer,milestones = milestones, gamma = 0.1)


    for epoch_num in range(num_of_epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0

        #training
        model.train()
        for idx,(data,target)  in enumerate(train_loader):
            
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            pred_output = model(data)

            loss = loss_criteria(pred_output, target)
            loss.backward()
            optimizer.step()
            

            train_epoch_loss += loss.item()
        
        print(scheduler.get_lr())
        scheduler.step()

        #validation
        model.eval()
        with torch.no_grad():

            for idx, (data,target) in enumerate(val_loader):

                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_criteria(output, target)
                val_loss = loss.item()
                val_epoch_loss += val_loss

        train_epoch_avg_loss = train_epoch_loss / len(train_loader)
        val_epoch_avg_loss = val_epoch_loss / len(val_loader)

        print(f"Epoch:[{epoch_num+1}/{num_of_epochs}] \t Train Loss:[{train_epoch_avg_loss:.6f}] \t Val Loss:[{val_epoch_avg_loss:.6f}]")
        train_val_loss_dict = {}
        train_val_loss_dict['train_epoch_avg_loss'] = train_epoch_avg_loss
        train_val_loss_dict['val_epoch_avg_loss'] = val_epoch_avg_loss
        tb.add_scalars('train vs val loss', train_val_loss_dict, epoch_num+1)

        #testing
        
        model.eval()
        with torch.no_grad():
            # cm = np.zeros((10,10)) + np.finfo(float).eps
            correct = 0
            total = 0
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # print(output.shape)
                # break

                pred = torch.argmax(output, dim = 1)
                
                target_np = target.cpu().numpy()
                pred_np = pred.cpu().numpy()

                total += len(target_np)

                correct += (target_np == pred_np).sum()

            perc_correct = (correct/total)*100
            tb.add_scalar(f'% age correct', perc_correct, epoch_num)

                # cm += confusion_matrix(target_np, pred_np, labels = list(np.arange(10)))

            # print(cm)

            # tp = np.diag(cm)
            # fp = np.sum(cm, axis = 0) - tp
            # fn = np.sum(cm, axis = 1) - tp
            # tn = np.sum(cm) - np.sum(tp) - np.sum(fp) + np.sum(fn)

            # print(f'Accuracy: {acc} %')
            # print(f'Mean Acc: {np.mean(acc)}')
            # print(f'Percentage Correct: {(np.sum(tp))/len(test_data)}')
            # print(f"sum of tp: {np.sum(tp)}")
            # print(f"sum of tn: {np.sum(tn)}")
            # print(f"test_data len : {len(test_data)}; test_data_loader len: {len(test_loader)}")
            # print(f"sum of cm elements: {np.sum(cm)}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CIFAR benchmarking")
    parser.add_argument('--lr',type=float, default = 0.001, help="learning_rate")
    parser.add_argument('--momentum',type=float, default = 0.9, help="momentum")
    parser.add_argument('--weight_decay',type=float, default = 0.0001, help="weight_decay")
    parser.add_argument('--train_batch_size', type=int, default = 512, help='training batch size')
    parser.add_argument('--validation_batch_size', type=int, default = 64, help='validation batch size')
    parser.add_argument('--test_batch_size', type=int, default = 1000, help='testing batch size')
    parser.add_argument('--epochs', type=int, default = 50,help="epoch_count")


    args = parser.parse_args()

    tb = SummaryWriter()
    pprint.pprint(vars(args))
    cifar_trainer(args, tb)
