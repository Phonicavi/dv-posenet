#!/usr/local/anaconda2/bin/python
from H5 import Input, Target


def h5_initialize():
	base_dir = "/Users/Phonic/Documents/G/Course/Graduate/Lab/Code/myDataset/"
	input_list = ["train_input.h5", "valid_input.h5"]
	target_list = ["train_target.h5", "valid_target.h5"]
	return (Input(base_dir, input_list[0]), Input(base_dir, input_list[1]), 
		Target(base_dir, target_list[0]), Target(base_dir, target_list[1]))


def dataset_process(_set):
	[ipt, tgt] = _set
	num = ipt.count()


if __name__ == "__main__":

	t_input, v_input, t_target, v_target = h5_initialize()

	train_set = [t_input, t_target]
	valid_set = [v_input, v_target]
	pairs = [train_set, valid_set]

	for _s in pairs:
		dataset_process(_s)
