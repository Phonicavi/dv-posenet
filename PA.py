from optparse import OptionParser


def parse_args():

	parser = OptionParser()
	parser.add_option("-d", "--dataset-path", type="string", dest="dir_data", 
		default="/home/yinger650/code/data/h36m/myDataset/")
	parser.add_option("-a", "--annot-path", type="string", dest="path_ann", 
		default="/home/yinger650/code/data/h36m/annot/")
	parser.add_option("-i", "--images-path", type="string", dest="path_img", 
		default="/home/yinger650/code/c2f/c2f-vol-train/data/h36m/images")
	parser.add_option("-m", "--marked-path", type="string", dest="path_img_marked", 
		default="/home/yinger650/code/data/h36m/images_marked/")
	parser.add_option("-w", "--thickness", type="int", dest="thick", 
		default=2)
	options, arguments = parser.parse_args()

	return options
