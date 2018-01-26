import sys
import numpy as np 
from PIL import Image
import pandas as pd 
import math

def read_images(n = 2000, start = 0, mode = "", filename = "class.jpg"):
	'''
	Find the pixel values for each pixel in each image;
	Store pixel values
	'''
	pixel_values = {}
	if mode != "":
		if mode == "background":
			path = "background/"
			path_write = "background_pixels/"
		if mode == "faces":
			path = "faces/face"
			path_write = "faces_pixels/"
		for i in range(start, n):
			image = path + str(i) + ".jpg"
			im = Image.open(image)
			im = im.convert("L")
			pixel_values[i] = find_pixel_values(im)
			df = pd.DataFrame(pixel_values[i])
			df.to_csv("./" + path_write + str(i) + ".csv", index = False)

	else:
		image = filename
		im = Image.open(image)
		im = im.convert("L")
		pixel_values[image] = find_pixel_values(im)
		df = pd.DataFrame(pixel_values[image])
		df.to_csv("./class.csv", index = False)
	return pixel_vlues


def find_pixel_values(image):
	'''
	Find the matrix of pixel values of an image
	'''
	pixel_v = []
	width, height = image.size
	for j in range(height):
		row_v = []
		for k in range(width):
			row_v.append(image.getpixel((k, j)))
		pixel_v.append(row_v)
	return pixel_v


def find_iimages(pixel_values, height = 64, width = 64, n = 2000, mode = ""):
	'''
	Find integral images and store them
	'''
	integrals = {}
	if mode == "faces":
		path_write = "./faces_integral/"
	elif mode == "background":
		path_write = "./background_integral/"
	else:
		path_write = "./"
	for i in pixel_values:
		integral = []
		#if (n == 1):
		#print(i)
		pv = np.array(pixel_values[i])
		for h in range(1, height + 1):
			integral_row = []
			for w in range(1, width + 1):
				print(h, w)
				curr_pv = pv[:h, :w]
				integral_row.append(np.sum(curr_pv))
			integral.append(integral_row)
		integrals[i] = integral
		df = pd.DataFrame(integrals[i])
		df.to_csv(path_write + str(i) + ".csv", index = False)
	return integrals


def find_features(height = 64, width = 64, step = 6):
	'''
	Find all possible features given the step
	'''
	count_h = height // step
	count_w = width // step
	features = []
	for h in range(width):
		for w in range(width):
			for ch in range(1, count_h + 1):
				for wh in range(1, count_w + 1):
					x0 = w
					y0 = h
					x1 = x0 + step * wh
					y1 = y0
					x2 = x0
					y2 = y0 + step * ch
					x3 = x1
					y3 = y2
					if (x3 < width) and (y3 < height):
						features.append([x0, y0, x1, y1, x2, y2, x3, y3])
	features = features + features # double the feature table so that both horizontal and vertical are considered
	df = pd.DataFrame(features)
	df.to_csv("./features.csv", index = False)
	return features


def select_features(features, num = 4000):
	'''
	select a subset of all features generated above
	Default number of features = 4000
	'''
	length = len(features)
	step = length // num
	selected_features = [i for i in range(length) if i % step == 0] # select features' ids instead
	df = pd.DataFrame(selected_features)
	df.to_csv("./selected_features.csv", index = False)
	return selected_features


def find_features_values(iimages, features_id, start = 0, n = 2000, mode = ""):
	'''
	Compute the value each feature in each images
	'''
	prefix = mode + str(n) +  "_"
	features_values = []

	for feature in features:
		print(feature)
		x0 = feature[0]
		y0 = feature[1]
		x1 = feature[2]
		y1 = feature[3]
		x2 = feature[4]
		y2 = feature[5]
		x3 = feature[6]
		y3 = feature[7]
		xa = (x0 + x1) // 2  # vertical mid point
		ya = y0
		xb = (x2 + x3) // 2
		yb = y2
		xc = x0             # horizontal mid point
		yc = (y0 + y2) // 2 
		xd = x1
		yd = (y1 + y3) // 2
		verticals = []
		horizontals = []
		for i in range(start, n):
			# first compute vertical 2-rectangles
			pv = iimages[i]
			s0 = pv[x0][y0]
			sa = pv[xa][ya]
			s2 = pv[x2][y2]
			sb = pv[xb][yb]
			black = s0 + sb- sa - s2
			s1 = pv[x1][y1]
			s3 = pv[x3][y3]
			white = sa + s3 - sb - s1
			vertical = black - white
			verticals.append(vertical)
			sc = pv[xc][yc]
			sd = pv[xd][yd]
			black = s0 + sd - sc - s1
			white = sc + s3 - s2 - sd
			horizontal = black - white
			horizontals.append(horizontal)
		features_values.append(verticals)
		features_values.append(horizontals)
	df = pd.DataFrame(features_values)
	df.to_csv("./" + prefix + "features_values.csv", index = False)
	return features_values


def combined_features(imfaces, imbackground):
	# combine the feature values from faces and background
	num = len(imfaces)
	features_values = [imfaces[i] + imbackground[i] for i in range(num)]
	return features_values


def find_is_face(numf, numb):
	# generate an array indicating weather an image is a face
	faces = [1] * numf
	background = [-1] * numb
	is_face = np.array(faces + background)
	return is_face


def find_threshold(features_values, is_face, weights = None, training_num = 400):
	'''
	the features_values here is the combined value
	'''
	if weights == None:
		weights = np.array([1] * training_num)
		weights = weights / training_num
	threshold = [] # dimension = number of features
	polarity = [] # dimension = number of features
	errors = []
	for i, values in enumerate(features_values):
		#print("threshold:", i)
		temp_v = sorted([(values[k], is_face[k], weights[k]) for k in range(training_num)])
		#print([v[0] for v in temp_vf])
		temp_polarity = None
		error = 1
		temp_threshold = 0
		for j in range(training_num-1):
			left = temp_v[:j+1]
			left_neg = sum([v[2] for v in left if v[1] == -1])
			left_pos = sum([v[2] for v in left if v[1] == 1])
			total_neg = sum([v[2] for v in temp_v if v[1] == -1])
			total_pos = sum([v[2] for v in temp_v if v[1] == 1])
			neg = left_pos + (total_neg - left_neg)
			pos = left_neg + (total_pos - left_pos)
			if min(pos, neg) < error:
				error = min(pos, neg)
				temp_threshold = j
				if pos <= neg:
					temp_polarity = "pos"
				else:				
					temp_polarity = "neg"
		t = (temp_v[temp_threshold][0] + temp_v[temp_threshold+1][0]) / 2
		threshold.append(t)
		polarity.append(temp_polarity)
		errors.append(error)
	return threshold, polarity, errors


def adaboost(features_values, is_face, termination, weights = None, training_num = 400):
	'''
	A strong learner is a list of dictionaries of the form
	[{"id": , "polarity": , "threshold": , "prediction": , "weights": r}]

	features_values is the faces200.csv file
	'''
	if weights == None:
		weights = np.array([1] * training_num)
		weights = weights / training_num
	it = 0
	learners = []
	length = len(features_values)
	used_features = []
	while (it < termination):
		print("iteration = ", it)
		learner = {}
		threshold, polarity, errors = find_threshold(features_values, is_face, weights, training_num = training_num)
		for feature in used_features:
			errors[feature] = 1
		error = min(errors)
		feature_id = errors.index(error)
		used_features.append(feature_id)
		prediction = []
		for value in features_values[feature_id]: # for each imagea
			if (value <= threshold[feature_id]) and (polarity[feature_id] == "neg"):
				prediction.append(-1)
			elif (value > threshold[feature_id]) and (polarity[feature_id] == "neg"):
				prediction.append(1)
			elif (value <= threshold[feature_id]) and (polarity[feature_id] == "pos"):
				prediction.append(1)
			elif (value > threshold[feature_id]) and (polarity[feature_id] == "pos"):
				prediction.append(-1)
		# Now update weights
		alpha = (1/2) * math.log((1 - error) / error)
		print("alpha = ", alpha)
		z = 2 * math.sqrt((error * (1 - error))) 
		prediction = np.array(prediction)
		temp = (-alpha) * (prediction * is_face)
		weights = (weights * np.exp(temp))/z
		learner["id"] = feature_id
		learner["threshold"] = threshold[feature_id]
		learner["polarity"] = polarity[feature_id]
		learner["weights"] = alpha
		learner["error"] = error
		learner["prediction"] = prediction
		learners.append(learner)
		# Take out used features
		it += 1
	return learners, weights


def test_adaboost(learners, fv, bv):
	print("first test on all faces")
	prediction = np.array([0] * 2000)
	for learner in learners:
		features_values = np.array(fv[learner["id"]])
		polarity = learner["polarity"]
		threshold = learner["threshold"]
		weights = learner["weights"]
		if polarity == "pos":
			faces = features_values <= threshold
			background = features_values > threshold
		if polarity == "neg":
			faces = features_values >= threshold
			background = features_values < threshold
		features_values[faces] = 1
		features_values[background] = -1
		single_p = features_values * weights
		prediction = prediction + single_p
	faces = prediction >= 0
	background = prediction < 0
	prediction[faces] = 1
	prediction[background] = 0
	num_faces = np.count_nonzero(prediction)
	print("recognize faces = ", num_faces)

	print("then test on all background")
	prediction = np.array([0] * 2000)
	for learner in learners:
		features_values = np.array(bv[learner["id"]])
		polarity = learner["polarity"]
		threshold = learner["threshold"]
		weights = learner["weights"]
		if polarity == "pos":
			faces = features_values <= threshold
			background = features_values > threshold
		if polarity == "neg":
			faces = features_values >= threshold
			background = features_values < threshold
		features_values[faces] = 1
		features_values[background] = -1
		single_p = features_values * weights
		prediction = prediction + single_p
	faces = prediction >= 0
	background = prediction < 0
	prediction[faces] = 0
	prediction[background] = 1
	num_background = np.count_nonzero(prediction)
	print("recognize background = ", num_background)


def test_photo_adaboost(learners, photo_pv, features):
	faces = []
	height = len(photo_pv) - 64
	#print(height)
	width = len(photo_pv[0]) - 64
	#print(width)
	for i in range(height):
		for j in range(width):
			#i = x * 64
			#j = y * 64
			print("i=", i, "j=", j)
			value = 0
			for learner in learners:
				iden = learner["id"]
				if iden < len(features):
					loc = features[iden]
					vertical = True
				else:
					loc = features[iden - len(features)]
					vertical = False
				threshold = learner["threshold"]
				weight = learner["weights"]
				polarity = learner["polarity"]
				x0 = i + loc[0]
				y0 = j + loc[1]
				x1 = i + loc[2]
				y1 = j + loc[3]
				x2 = i + loc[4]
				y2 = j + loc[5]
				x3 = i + loc[6]
				y3 = j + loc[7]
				#print("x0 = ", x0, "y0 = ", y0)
				s0 = photo_pv[x0][y0]
				#print("s0 = ", s0)
				s1 = photo_pv[x1][y1]
				s2 = photo_pv[x2][y2]
				s3 = photo_pv[x3][y3]
				if vertical:
					xa = (x0 + x1) // 2
					ya = y0
					xb = (x2 + x3) // 2
					yb = y2
					sa = photo_pv[xa][ya]
					sb = photo_pv[xb][yb]
					black = s0 + sb - sa - s2
					white = sa + s3 - s1 - sb
				else:
					xa = x0
					ya = (y0 + y2) // 2
					xb = x1
					yb = (y1 + y3) // 2
					sa = photo_pv[xa][ya]
					sb = photo_pv[xb][yb]
					black = s0 + sb - s1 - sa
					white = sa + s3 - sb - s2
				fv = black - white
				#print("threshold = ", threshold)
				#print("polarity = ", polarity)
				#print("fv = ", fv)
				if (fv <= threshold) and (polarity == "neg"):
					pred = -1
				elif (fv > threshold) and (polarity == "neg"):
					pred = 1					
				elif (fv <= threshold) and (polarity == "pos"):
					pred = 1
				elif (fv > threshold) and (polarity == "pos"):
					pred = -1
				#print("prediction = ", pred)
				value += pred * weight
			#print(value)
			if value > 0:
				#print("positive value")
				faces.append((i, j))
			#print("--------------------------------------")
	return faces

def adjust_threshold(strong_learner, is_face, training_num = 400):
	prediction = np.array([0] * training_num)
	for learner in strong_learner:
		prediction = prediction +  learner["prediction"] * learner["weights"]
	sorted_values = sorted([(prediction[i], is_face[i]) for i in range(training_num)])
	for j in range(len(sorted_values)):
		if sorted_values[j][1] == 1:
			threshold = j
			print("threshold index = ", threshold)
			break
	threshold = (sorted_values[threshold - 1][0] + sorted_values[threshold][0])/2
	return threshold


def cascade(features_values, is_face, weights = None, termination = 5, training_num = 400):
	i = 0
	strong_learners = []
	while (i < termination):
		print("casecading", i+1)
		strong_learner, weights = adaboost(features_values, is_face, 3, weights = weights, training_num = training_num)
		new_threshold = adjust_threshold(strong_learner, is_face, training_num = training_num)
		print("threshold = ", new_threshold)
		#strong_learners.append((strong_learner, new_threshold))
		prediction = np.array([0] * training_num)
		for learner in strong_learner:
			learner["big_threshold"] = new_threshold
			p = (learner["prediction"] * learner["weights"])
			#print(p)
			prediction = prediction + p
		#print(prediction)
		strong_learners.append(strong_learner)
		face = prediction > new_threshold
		#print(face)
		temp = (np.array(features_values))[:,face]
		features_values = temp.tolist()
		training_num = len(features_values[0])
		print(training_num)
		is_face = ((np.array(is_face))[face]).tolist()
		weights = weights[face]
		total_weight = sum(weights)
		#print("old weights", weights)
		#print("total = ", total_weight)
		weights = weights / total_weight
		#print("new weights", weights)
		#aprint(sum(weights))
		i += 1
	return strong_learners

def test_one_cascade(learners, fv, bv, faces = True):
	if faces:
		cv = fv
	else:
		cv = bv
	prediction = np.array([0] * len(cv[0]))
	for learner in learners:
		fid = learner["id"]
		features_values = np.array(cv[learner["id"]])
		polarity = learner["polarity"]
		threshold = learner["threshold"]
		weights = learner["weights"]
		if polarity == "pos":
			faces = features_values <= threshold
			background = features_values > threshold
		if polarity == "neg":
			faces = features_values >= threshold
			background = features_values < threshold
		features_values[faces] = 1
		features_values[background] = -1
		single_p = features_values * weights
		prediction = prediction + single_p
	faces = prediction > learner["big_threshold"]
	background = prediction <= learner["big_threshold"]
	prediction[faces] = 1
	prediction[background] = 0
	num_faces = np.count_nonzero(prediction)
	print("number of faces recognized", num_faces)
	cv = (np.array(cv))[:, faces]
	cv = cv.tolist()
	print("number of images remaining", len(cv[0]))
	return cv


def test_cascade(cascaded_learners, fv, bv, faces = True):
	if faces:
		print("first test faces")
	else:
		print("Now test background")
	for learners in cascaded_learners:
		cv = test_one_cascade(learners, fv, bv, faces = faces)
		if faces:
			fv = cv
		else:
			bv = cv

'''	
if __name__ == "__main__":
	n = int(sys.argv[1])
	mode = sys.argv[2]
	read_images(n, mode)
'''
