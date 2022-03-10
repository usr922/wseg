
class Registry(object):
	def __init__(self, name):
		super(Registry, self).__init__()
		self._name = name
		self._module_dict = dict()
	
	@property
	def name(self):
		return self._name

	@property
	def module_dict(self):
		return self._module_dict

	def __len__(self):
		return len(self.module_dict)	

	def get(self, key):
		return self._module_dict[key]

	def register_module(self, module=None):
		if module is None:
			raise TypeError('fail to register None in Registry {}'.format(self.name))
		module_name = module.__name__
		if module_name in self._module_dict:
			raise KeyError('{} is already registry in Registry {}'.format(module_name, self.name))
		self._module_dict[module_name] = module
		return module

DATASETS = Registry('dataset')
BACKBONES = Registry('backbone')
NETS = Registry('nets')
