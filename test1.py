import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # descargar el dataset de entrenamiento
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2) # carga el dataset de entrenamiento

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # descargar el dataset de prueba

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2) # carga el dataset de prueba


classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imgshow(img):
	img = img / 2 + 0.5 # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
		
if __name__ == '__main__':
	net = Net()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(2):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data


			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')

	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


	########
	model=torch.load('imagenes_test3.pt')
	
	
	
	fakeLabels = []
	outputs = net(model[-1]) # transpose(1, 3, 32, 32) funciona en el net
	_, predicted = torch.max(outputs, 1)
	print('Predicted:', ' '.join('%5s' % classes[predicted[j]] for j in range(64)))
	for j in range(64):
		fakeLabels.append(predicted[j])
	
	
	# newimg, newlabel = np.random.rand(1, 32, 32, 3), 1 #<- dimension de cifar10
	
	
	#intente tomar una imagen de model[-1], 
	#converti el tensor de esa imagen en un numpy array, 
	#luego le di las dimensiones para que se pareciera a los datos de cifar10, movi las filas para que calzara con los inputs con transpose
	# lo agrege al trainloader junto con las etiquetas q se predijieron.

	#en teoria funciona, lo malo esq no se si esta bien el orden de las filas de los tensores.
	for j in range(0,len(model[-1])-1): #se demora un monton en ejecutar este for
		model1= torch.tensor((np.transpose(np.array(model[-1][j]).reshape(1, 3, 32, 32), (0,2 ,3 ,1)))) 
		trainset.data = np.r_[trainloader.dataset.data, model1] # se debe agregar las 64 imagenes <- iterear eso 64 veces.
	
	trainset.targets.append(fakeLabels)
	imgshow(torchvision.utils.make_grid(model[-1]))

	# fakeLabels = []
	# for elem in fakeList:
	#     output = cnn(fakeList[0])
	#     _, predicted = torch.max(outputs, 1)
	#     fakeLabels.append(predicted)
	
	


		
