import time
import torch
from tqdm import tqdm

def single_gpu_test(model, dataloader, prepare_func, inference_func, collect_func, save_step_func=None):
	model.eval()
	n_gpus = torch.cuda.device_count()
	#assert n_gpus == 1
	collect_list = []
	total_num = len(dataloader)
	with tqdm(total=total_num) as pbar:
		with torch.no_grad():
			for i_batch, sample in enumerate(dataloader):
				name = sample['name']
				image_msf = prepare_func(sample)
				result_list = []
				for img in image_msf:
					result = inference_func(model, img.cuda())	
					result_list.append(result)
				result_item = collect_func(result_list, sample)
				result_sample = {'predict': result_item, 'name':name[0]}
				#print('%d/%d'%(i_batch,len(dataloader)))
				pbar.set_description('Processing')
				pbar.update(1)
				time.sleep(0.001)

				if save_step_func is not None:
					save_step_func(result_sample)
				else:
					collect_list.append(result_sample)
	return collect_list
