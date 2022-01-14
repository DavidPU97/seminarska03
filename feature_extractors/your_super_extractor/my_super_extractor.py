import cv2, sys
from skimage import feature
import numpy as np
from skimage.feature import local_binary_pattern, daisy, hog, BRIEF
from skimage import filters


class LBP:
	def __init__(self, resize=100):
		self.resize=resize
		self.radius = 3
		self.n_points = 8 * self.radius

	def lbp(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		img = local_binary_pattern(img, self.n_points, self.radius)

		img = img.ravel()
		
		return img

	def lbp_hist(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		img = local_binary_pattern(img, self.n_points, self.radius)

		global_feature_vector = []

		for w in range(0, 100, 20):
			for h in range(0, 100, 20):
				global_feature_vector += np.histogram(img[w:w + 20, h:h + 20], bins=8)[0].tolist()

		global_feature_vector_norm = np.array(global_feature_vector) / np.linalg.norm(global_feature_vector)

		return global_feature_vector_norm.tolist()

	def sobel(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		img = filters.sobel(img)	#same as lbp (11.965811965811966)

		img = img.ravel()

		return img

	def canny(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		img = feature.canny(img) #(6.41025641025641)

		img = img.ravel()

		return img

	def daisy(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		img = daisy(img)  #(14.957264957264957)

		img = img.ravel()

		return img

	def hog(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		img_hog = hog(img) #(19.230769230769234)

		# img = img.ravel()

		return img_hog

	def hog_lbp(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		img_hog = hog(img)
		img_hog = img_hog.reshape(90, 90)

		img_hog_lbp = local_binary_pattern(img_hog, self.n_points, self.radius, method='uniform')

		global_feature_vector = []

		for w in range(0, 100, 20):
			for h in range(0, 100, 20):
				global_feature_vector += np.histogram(img_hog_lbp[w:w + 20, h:h + 20], bins=8)[0].tolist()

		global_feature_vector_norm = np.array(global_feature_vector) / np.linalg.norm(global_feature_vector)

		return global_feature_vector_norm.tolist()

	def lbp_hog(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))

		img_lbp = local_binary_pattern(img, self.n_points, self.radius)

		global_feature_vector = []

		for w in range(0, 100, 20):
			for h in range(0, 100, 20):
				global_feature_vector += np.histogram(img_lbp[w:w + 20, h:h + 20], bins=8)[0].tolist()

		global_feature_vector_norm = np.array(global_feature_vector) / np.linalg.norm(global_feature_vector)
		global_feature_vector_norm = global_feature_vector_norm.tolist()

		img_hog_lbp = hog(cv2.resize(np.array(global_feature_vector_norm), (100, 100)))
		# img_hog_lbp = img_hog.reshape(90, 90)
		return img_hog_lbp


if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = Extractor()
	features = extractor.extract(img)
	print(features)