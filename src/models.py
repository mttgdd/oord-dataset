import torch

def get_nn(name):
	if name == 'mobilenet_v2':
		return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
	elif name == 'googlenet':
		return torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
	elif name == 'alexnet':
		return torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
	elif name == 'vgg19':
		return torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
	elif name == 'cosplace_resnet101_1024':
		return torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet101", fc_output_dim=1024)
	elif name == 'cosplace_resnet18_512':
		return torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet18", fc_output_dim=512)
	elif name == 'cosplace_resnet50_2048':
		return torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
	elif name == 'cosplace_resnet152_2048':
		return torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet152", fc_output_dim=2048)
	elif name == 'cosplace_vgg16_512':
		return torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="VGG16", fc_output_dim=512)
