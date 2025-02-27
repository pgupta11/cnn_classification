import sys
sys.path.append('/Users/prachigupta/cbp/cnn_classification')
print(sys.path)
import torch
import torch.nn as nn
import torch.optim as optim
from cnn_classification.data.dataset import get_datasets, get_num_classes
from models.cnn import ConvNeuralNet
from utils.preprocess import normalize_data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
dataset_train, dataset_val = get_datasets()
num_classes = get_num_classes(dataset_train)

# Preprocess and normalize data
inputs_train = normalize_data(dataset_train)
inputs_val = normalize_data(dataset_val)
batch_size = 64
#loader
dloader_train = torch.utils.data.DataLoader(
    inputs_train, batch_size=batch_size, shuffle=True
)
dloader_val = torch.utils.data.DataLoader(
    inputs_val, batch_size=batch_size, shuffle=False
)
# Initialize model, loss function, and optimizer
model = ConvNeuralNet(num_classes).to(device)
loss_func = nn.CrossEntropyLoss()
lr = 0.008
optimizer = optim.SGD(model.parameters(), lr=lr)

# train and validate the network
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
	# load in the data in batches
    for i, (images, labels) in enumerate(dloader_train):  
        # move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # forward propagation
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        # backward propagation and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # at end of epoch check validation loss and accuracy on validation set
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        all_val_loss = []
        for images, labels in dloader_val:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            # calculate predictions
            predicted = torch.argmax(outputs, dim=1)
            # calculate actual values
            correct += (predicted == labels).sum().item()
            # calculate the loss
            all_val_loss.append(loss_func(outputs, labels).item())
        # calculate val-loss
        mean_val_loss = sum(all_val_loss) / len(all_val_loss)
        # calculate val-accuracy
        mean_val_acc = 100 * (correct / total)

    print(
        'Epoch [{}/{}], Loss: {:.4f}, Val-loss: {:.4f}, Val-acc: {:.1f}%'.format(
            epoch+1, num_epochs, loss.item(), mean_val_loss, mean_val_acc
        )
    )
torch.save(model,'cnn.pt')