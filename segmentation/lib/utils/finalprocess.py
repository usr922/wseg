import os

def writelog(cfg, period, metric=None, commit=''):
	filepath = os.path.join(cfg.ROOT_DIR,'log','logfile.txt')
	logfile = open(filepath,'a')
	import time
	logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	logfile.write('\t%s\n'%period)
	para_data_dict = {}
	para_model_dict = {}
	para_train_dict = {}
	para_test_dict = {}
	para_name = dir(cfg)
	for name in para_name:
		if 'DATA_' in name:
			v = getattr(cfg,name)
			para_data_dict[name] = v
		elif 'MODEL_' in name:
			v = getattr(cfg,name)
			para_model_dict[name] = v
		elif 'TRAIN_' in name:
			v = getattr(cfg,name)
			para_train_dict[name] = v
		elif 'TEST_' in name:
			v = getattr(cfg,name)
			para_test_dict[name] = v
	writedict(logfile, {'EXP_NAME': cfg.EXP_NAME})
	writedict(logfile, para_data_dict)
	writedict(logfile, para_model_dict)
	if 'train' in period:
		writedict(logfile, para_train_dict)
	else:
		writedict(logfile, para_test_dict)
		writedict(logfile, metric)

	logfile.write(commit)
	logfile.write('=====================================\n')
	logfile.close()

def writedict(file, dictionary):
	s = ''
	for key in dictionary.keys():
		sub = '%s:%s  '%(key, dictionary[key])
		s += sub
	s += '\n'
	file.write(s)

