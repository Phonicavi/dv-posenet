from optparse import OptionParser


def parse_args():

	parser = OptionParser()
	parser.add_option("-d", "--dataset-path", type="string", dest="dir_data", 
		default="/Users/Phonic/Documents/G/Course/Graduate/Lab/Code/myDataset/")
	parser.add_option("-a", "--annot-path", type="string", dest="path_ann", 
		default="/Users/Phonic/Documents/G/Course/Graduate/Lab/Code/h36m/annot/")
	parser.add_option("-i", "--images-path", type="string", dest="path_img", 
		default="/Users/Phonic/Documents/G/Course/Graduate/Lab/Code/h36m/images/")
	parser.add_option("-m", "--marked-path", type="string", dest="path_img_marked", 
		default="/Users/Phonic/Documents/G/Course/Graduate/Lab/Code/h36m/images_marked/")
	parser.add_option("-w", "--thickness", type="int", dest="thick", 
		default=2)
	options, arguments = parser.parse_args()

	return options
