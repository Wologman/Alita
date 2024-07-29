import torch
gpu = torch.cuda.is_available()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if gpu else 'CPU'))
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_summary(device=None, abbreviated=False))

import timm
listofmodels = timm.list_models(pretrained=True)
print(listofmodels)