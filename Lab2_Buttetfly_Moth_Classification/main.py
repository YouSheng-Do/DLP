import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from VGG19 import VGG19
from ResNet50 import ResNet50
from dataloader import BufferflyMothLoader


def evaluate(model, dataloader, loss_function, device):
    num_correct = 0
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # print(images)
            # print(outputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predict = torch.max(outputs, 1)
            # print(predict)
            # print(labels)
            num_correct += (predict == labels).sum().item()
            # print(num_correct)
    
    val_loss = val_loss / len(dataloader.dataset)
    val_acc = 100 * num_correct / len(dataloader.dataset)

    return val_loss, val_acc

def test(model, dataloader, device):
    model.eval()

    num_correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs, 1)
            num_correct += (predict == labels).sum().item()
    
    acc = 100 * num_correct / len(dataloader.dataset)

    return acc

def train(model, train_dataloader, loss_function, optimizer, device):
    running_loss = 0.0
    # num_correct = 0
    model.train()
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # _, predict = torch.max(outputs, 1)
        # num_correct += (predict == labels).sum().item()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    # epoch_loss = running_loss / len(train_dataloader.dataset)
    # acc = 100 * num_correct / len(train_dataloader.dataset)

    # return epoch_loss, acc
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    
    mode = args.mode

    device = torch.device("cuda")

    vgg19 = VGG19().to(device)
    resnet50 = ResNet50().to(device)
   
        
    loss_function = nn.CrossEntropyLoss()

    vgg19_optimizer = optim.SGD(vgg19.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
    resnet50_optimizer = optim.SGD(resnet50.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    
    vgg19_scheduler = optim.lr_scheduler.ReduceLROnPlateau(vgg19_optimizer, factor=0.1, patience=10, verbose=True)
    resnet50_scheduler = optim.lr_scheduler.ReduceLROnPlateau(resnet50_optimizer, factor=0.1, patience=10, verbose=True)

    train_dataset = BufferflyMothLoader(root='dataset', mode='train')    
    val_dataset = BufferflyMothLoader(root='dataset', mode='eval')
    test_dataset = BufferflyMothLoader(root='dataset', mode='test')
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)

    if mode == 'train':
        vgg19_train_acc_history = []
        vgg19_test_acc_history = []
        best_acc = 0.0
        num_epochs = 100
        print("Start training VGG19...")
        for epoch in range(num_epochs):
            train(vgg19, train_dataloader, loss_function, vgg19_optimizer, device)
            train_loss, train_acc = evaluate(vgg19, train_dataloader, loss_function, device)
            val_loss, val_acc = evaluate(vgg19, val_dataloader, loss_function, device)
            test_loss, test_acc = evaluate(vgg19, test_dataloader, loss_function, device)

            vgg19_scheduler.step(val_loss)

            vgg19_test_acc_history.append(test_acc)
            vgg19_train_acc_history.append(train_acc)
            
            print(f'(VGG19) Epoch: {epoch} Train Loss: {train_loss} Train Accuracy: {train_acc:.1f}% Validation Loss: {val_loss} Validation Accuracy: {val_acc}% Test Loss: {test_loss} Test Accuracy: {test_acc}%')
                
            if val_acc > best_acc:
                print(f'(VGG19) Epoch: {epoch} Saving best model with validation accuracy {val_acc}%')
                best_acc = val_acc
                torch.save(vgg19.state_dict(), 'best_vgg19.pth')    

        resnet50_train_acc_history = []
        resnet50_test_acc_history = []
        best_acc = 0.0
        num_epochs = 100
        print("Start training ReNet50...")
        for epoch in range(num_epochs):
            train(resnet50, train_dataloader, loss_function, resnet50_optimizer, device)
            train_loss, train_acc = evaluate(resnet50, train_dataloader, loss_function, device)
            val_loss, val_acc = evaluate(resnet50, val_dataloader, loss_function, device)
            test_loss, test_acc = evaluate(resnet50, test_dataloader, loss_function, device)

            resnet50_scheduler.step(val_loss)

            resnet50_test_acc_history.append(test_acc)
            resnet50_train_acc_history.append(train_acc)
            
            print(f'(ResNet50) Epoch: {epoch} Train Loss: {train_loss} Train Accuracy: {train_acc:.1f}% Validation Loss: {val_loss} Validation Accuracy: {val_acc}% Test Loss: {test_loss} Test Accuracy: {test_acc}%')
                
            if val_acc > best_acc:
                print(f'(ResNet50) Epoch: {epoch} Saving best model with validation accuracy {val_acc}%')
                best_acc = val_acc
                torch.save(resnet50.state_dict(), 'best_resnet50.pth')

        plt.plot(vgg19_train_acc_history, label='VGG19_train_acc')
        plt.plot(vgg19_test_acc_history, label='VGG19_test_acc')       
        plt.plot(resnet50_train_acc_history, label='ResNet50_train_acc')
        plt.plot(resnet50_test_acc_history, label='ResNet50_test_acc')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig('accuracy_comparision.png')
        plt.show()
                
    elif mode == 'test':
        vgg19.load_state_dict(torch.load('best_vgg19.pth'))
        train_acc = test(vgg19, train_dataloader, device)    
        test_acc = test(vgg19, test_dataloader, device)
        val_acc = test(vgg19, val_dataloader, device)
        # print(val_acc)
        print('----------VGG19----------')
        print(f'VGG19        |   Train accuracy: {train_acc:.2f}%|   Test accracy: {test_acc:.2f}%')
        resnet50.load_state_dict(torch.load('best_resnet50.pth'))
        train_acc = test(resnet50, train_dataloader, device)
        test_acc = test(resnet50, test_dataloader, device)
        val_acc = test(resnet50, val_dataloader, device)
        # print(val_acc)
        print('----------ResNet50----------')
        print(f'Resnet50        |   Train accuracy: {train_acc:.2f}%|   Test accracy: {test_acc:.2f}%')
                