import numpy as np

def accuracy(output, target, topk=(1,), debug=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        #res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(1.0 / batch_size))
    
    return res


def accuracy_top1(outputs, labels, debug=False):

	batch_size = len(outputs)
	res = np.zeros(batch_size, dtype=int)
	for i in range(batch_size):
		output = outputs[i] #.detach().cpu().numpy()
		#label = int(labels[i])
		label = np.argmax(labels[i])
		predict = np.argmax(output)
		res[i] = 1 if label==predict else 0
		if debug: print('i={}: res={} (label={}, predict={})'.format(i, res[i], label, predict))
	return np.mean(res)


def accuracy_topk(outputs, labels, k=1, debug=False):

	batch_size = len(outputs)
	res = np.zeros(batch_size, dtype=int)
	for i in range(batch_size):
		output = outputs[i] #.detach().cpu().numpy()
		#label = int(labels[i])
		label = np.argmax(labels[i])
		#predict = np.argmax(output)
		topk_predicts = output.argsort()[::-1][:k]
		res[i] = 1 if label in set(topk_predicts) else 0
		if debug: print('i={}: res={} (label={}, predicts={})'.format(i, res[i], label, list(topk_predicts)))

	return np.mean(res)