import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import PIL
from PIL import Image, ImageFile
import timm
ImageFile.LOAD_TRUNCATED_IMAGES = True


train_data_path = "D:\\train"
val_data_path = "D:\\val"
test_data_path = "D:\\test"


class Dataset(object):
	def __getitem__(self, index):
		raise NotImplementedError
	def __len__(self):
		raise NotImplementedError

transforms = transforms.Compose([transforms.Resize([224,224],interpolation=PIL.Image.BICUBIC),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms)
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=transforms)
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=transforms)

batch_size=128

train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)



class SwTr(nn.Module):
	def __init__(self,x, num_classes=3):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, num_classes)
		)
	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x

if (torch.cuda.is_available()):
	device ='cuda'
else:
	device ='cpu'

y=torch.rand(2)

#Initializing a class  simpl
simpl = SwTr(y)
print("Do you want to re-train the model (0) or use ready-made weights and parameters(1)? ")

b = input()
a = int(b)

if a>0:
	simpl.load_state_dict(torch.load('D:\\swin_transformer_model.pth',weights_only=True))
	simpl.eval() 
	simpl.to(device)

optimizer = optim.Adam(simpl.parameters(), lr=0.001)

def train(model, optimizer, loss_fn, train_loader, val_loader, device):
	epochs = 10
	train_iterator =0
	valid_iterator =0
	num_correct = 0
	num_examples = 0
	for epoch in range(epochs):
		training_loss = 0.0
		valid_loss = 0.0
		model.train()
		if (torch.cuda.is_available()):
			model.cuda()

		for batch in train_loader:
			optimizer.zero_grad()
			inputs, targets = batch
			inputs = inputs.to(device)
			targets = targets.to(device)
			output = model(inputs)
			loss = loss_fn(output, targets)
			loss.backward()
			optimizer.step()
			training_loss += loss.data.item()
			train_iterator+=1
		training_loss /= train_iterator

		model.eval()
		if (torch.cuda.is_available()):
			model.cuda()
		with torch.no_grad():
			for batch in val_loader:
				inputs, targets = batch
				inputs = inputs.to(device)
				output = model(inputs)
				targets = targets.to(device)
				loss = loss_fn(output,targets)
				valid_loss += loss.data.item()
				correct = torch.eq(torch.max(F.log_softmax(output,1),1)[1], targets).view(-1)
				num_correct += torch.sum(correct).item()
				num_examples += correct.shape[0]
				valid_iterator+=1
			valid_loss /= valid_iterator
		print('E: {:2}, Training Loss: {:.10f}, Validation Loss: {:.10f}, accuracy = {:.2f}, {:}/{:}'.format(epoch+1, training_loss, valid_loss, num_correct / num_examples, num_correct, num_examples))

	torch.save(model.state_dict(), 'D:\\swin_transformer_model.pth')
	print("Model has been saved!")



def test(model, test_loader, device):
	num_correct = 0
	num_examples = 0
	if (torch.cuda.is_available()):
			model.cuda()
	with torch.no_grad():
		for batch in test_loader:
			inputs, targets = batch
			inputs = inputs.to(device)
			output = model(inputs)
			targets = targets.to(device)
			correct = torch.eq(torch.max(F.log_softmax(output, 1), 1)[1], targets).view(-1)
			num_correct += torch.sum(correct).item()
			num_examples += correct.shape[0]
			accuracy = num_correct / num_examples
		print(f'Accuracy on test set: {accuracy:.2f}, {num_correct}/{num_examples}')




if a>0:
	test(simpl, test_data_loader, device)
else:
	train(simpl, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, device)

	

labels = ['blue','green','red']

img1 = Image.open('D:\\blue1.png').convert('RGB')			 
img1 = transforms(img1)
img1 = img1.unsqueeze(0)

img2 = Image.open('D:\\green1.png').convert('RGB')			 
img2 = transforms(img2)
img2 = img2.unsqueeze(0)

img3 = Image.open('D:\\red1.png').convert('RGB')			 
img3 = transforms(img3)
img3 = img3.unsqueeze(0)

if (torch.cuda.is_available()):
	img1 = img1.cuda()
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

if (torch.cuda.is_available()):
	img2 = img2.cuda()
prediction = simpl(img2)
prediction = prediction.argmax()
print(labels[prediction])

if (torch.cuda.is_available()):
	img3 = img3.cuda()
prediction = simpl(img3)
prediction = prediction.argmax()
print(labels[prediction])
