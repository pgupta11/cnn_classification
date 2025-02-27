import torch
import matplotlib.pyplot as plt
from data.dataset import get_datasets
from models.cnn import ConvNeuralNet
from utils.preprocess import preprocess_data
from tqdm.auto import tqdm
from utils.preprocess import normalize_data
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
_, dataset_test = get_datasets()

# Preprocess data

inputs_test = normalize_data(dataset_test)

print(len(inputs_test))
print(inputs_test[0][0].shape)
# Load model
num_classes = 10  # Adjust accordingly
model = torch.load('cnn.pt',weights_only=False)
# switch to evaluation mode and device
model.eval().to(device)
print(device)
inputs_test_tensor = [item[0] for item in inputs_test[:10]]
stacked = torch.stack(inputs_test_tensor, dim=0)
print(stacked.shape)

outputs = model(stacked)
#outputs are the logits for each class. 
#The model(input_tensors) produces output logits, which are typically a tensor with shape [batch_size, num_classes]
#so each row of the output tensor is the model's prediction for one example.
#The predicted class is the class with the highest score for each example.
#torch.argmax is selecting the predicted class for each input by picking the index of the highest probability score (or logit).
print(outputs.shape)
predictions = torch.argmax(outputs, dim=1)
# Display some results
for i, image in enumerate(dataset_test['img'][:10]):
    plt.imshow(image)
    plt.show()
    print(dataset_test.features['label'].names[predictions[i]])