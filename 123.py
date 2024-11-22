import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import PIL
from PIL import Image


train_data_path = "Q:\\train"
val_data_path = "Q:\\val"
test_data_path = "Q:\\test"


class Dataset(object):
	def __getitem__(self, index):
		raise NotImplementedError
	def __len__(self):
		raise NotImplementedError

transforms = transforms.Compose([transforms.Resize(64),	transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms)
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=transforms)
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=transforms)

batch_size=64

train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)



class Simpl(nn.Module):

	def __init__(self,x):
		super().__init__()
		self.fc1 = nn.Linear(12288, 84)
		self.fc2 = nn.Linear(84, 50)
		self.fc3 = nn.Linear(50,2)

	def forward(self,x):
		x = x.view(-1, 12288)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

y=torch.rand(2)
simpl = Simpl(y)


optimizer = optim.Adam(simpl.parameters(), lr=0.001)

training_loss =0.0
valid_loss =0.0

def train(model, optimizer, loss_fn, train_loader, val_loader, device="cpu"):
	epochs = 70
	train_iterator =0
	valid_iterator =0
	for epoch in range(epochs):
		training_loss = 0.0
		valid_loss = 0.0
		model.train()
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
		num_correct = 0
		num_examples = 0
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
		print('Epoch: {}, Training Loss: {:.10f}, Validation Loss: {:.10f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))


device ='cpu'
train(simpl, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, test_data_loader, device)


labels = ['circle','triangle']


##########################################
img1 = Image.open('D:\\Krug.jpg')			 
img1 = transforms(img1)
img1 = img1.unsqueeze(0)


img2 = Image.open('D:\\Treugolnik.jpg')			 
img2 = transforms(img2)
img2 = img2.unsqueeze(0)


#circle
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#triangle
prediction = simpl(img2)
prediction = prediction.argmax()
print(labels[prediction])

#circle
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#circle
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#circle
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#circle
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#circle
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])



'''
##########################################
img1 = Image.open('D:\\blueg.jpg')			 
img1 = transforms(img1)
img1 = img1.unsqueeze(0)

#///////////////////////////////////////
img2 = Image.open('D:\\redg.jpg')			 
img2 = transforms(img2)
img2 = img2.unsqueeze(0)

#*****************************************
img3 = Image.open('D:\\greeng.jpg')			 
img3 = transforms(img3)
img3 = img3.unsqueeze(0)

#Custom
img4 = Image.open('D:\\17.jpg')			 
img4 = transforms(img4)
img4 = img4.unsqueeze(0)

#blue
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#space
prediction = simpl(img4)
prediction = prediction.argmax()
print(labels[prediction])

#red
prediction = simpl(img2)
prediction = prediction.argmax()
print(labels[prediction])

#green
prediction = simpl(img3)
prediction = prediction.argmax()
print(labels[prediction])

#blue
prediction = simpl(img1)
prediction = prediction.argmax()
print(labels[prediction])

#space
prediction = simpl(img4)
prediction = prediction.argmax()
print(labels[prediction])

#green
prediction = simpl(img3)
prediction = prediction.argmax()
print(labels[prediction])

#red
prediction = simpl(img2)
prediction = prediction.argmax()
print(labels[prediction])

#green
prediction = simpl(img3)
prediction = prediction.argmax()
print(labels[prediction])

#green
prediction = simpl(img3)
prediction = prediction.argmax()
print(labels[prediction])

#space
prediction = simpl(img4)
prediction = prediction.argmax()
print(labels[prediction])

#red
prediction = simpl(img2)
prediction = prediction.argmax()
print(labels[prediction])

#space
prediction = simpl(img4)
prediction = prediction.argmax()
print(labels[prediction])
'''