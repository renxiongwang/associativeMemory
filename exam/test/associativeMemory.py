from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def transform(img):
	result = np.init_(img.reshape((-1, 1)))
	length = len(result)
	for i in range(length):
		if result[i] == 0:
			result[i] = -1
		else:
			result[i] = 1
	return result

def training(W, img, al):
	p = transform(img)
	plen = len(p)
	if W is None:
		W = np.zeros((plen, plen))
	t = p
	result = W + al * np.dot(p, t.T)
	return result

def hardlim(a):
	a[ a >= 0 ] = 1
	a[ a < 0 ] = 0
	return a

def testing(W, img):
	result = hardlim(np.dot(transforming(img).T, W))
	return result

def main():
	alpha = 0.5
	trainDir = "train/"
	testDir = "test/"
	trainImg = os.listdir(trainDir)
	trainImg.sort()
	
	trnLen = len(trainImg)
	W = None;
	
	for i in range(trnLen):
		img = np.array(Image.open(trainDir + trainImg[i]))
		W = training(W, img, alpha)
	count = 1
	for i in range(trnLen):
		img = Image.open(trainDir + trainImg[i])
		plt.subplot(1, trnLen, i + 1)
		plt.title(str(i))
		plt.imshow(img, cmap = plt.cm.gray)
	
	dirsAll = os.listdir(testDir)
	dirsAll.sort()
	dirLen = len(dirsAll)
	count = 1

	plt.figure(figsize = (10, 10))

	for i in range(dirLen):
		files = os.listdir(testDir + dirsAll[i])
		files.sort()
		filesLen = len(files)
		for j in range(filesLen):
			fullPath = testDir + dirsAll[i] + '/' + files[j]
			img = np.array(Image.open(fullPath))
			
			imageResult = testing(W, img)
			DisSource = hardlim(transform(img).reshape(img.shape))
			DisResult = imageResult.reshape(img.shape)
			
			plt.subplot(dirLen, 2 * filesLen, count)
			plt.title('source')
			count = count + 1
			plt.imshow(DisSource, cmp = plt.cm.gray)

			plt.subplot(dirLen, 2 * filesLen, count)
			plt.title('result')
			count = count + 1
			plt.imshow(DisResult, cmap = plt.cm.gray)

	fig = plt.get_current_fig_manager()
	fig.window.wm_geometry("+100+0")
	plt.show()
