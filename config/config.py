import yaml
import argparse

# TODO: reuse boilerplate

def get_cfg_impl(config_file):
	with open(config_file) as f:
		cfg = yaml.load(f, Loader=yaml.FullLoader)
	return cfg

def get_data_yaml(yaml_file):
	with open(yaml_file) as f:
		data = yaml.load(f, Loader=yaml.FullLoader)
	return data['data']

def get_gps_yaml(yaml_file):
	with open(yaml_file) as f:
		gps = yaml.load(f, Loader=yaml.FullLoader)
	return gps

def get_cfg():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)

	args = parser.parse_args()

	return get_cfg_impl(args.config)