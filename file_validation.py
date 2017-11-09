import os
import pickle


def load_names(base_path, inventory):
    dump_path = "./%s.pickle_list" % inventory[:-4]
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path, "rb"))
    res = []
    with open(os.path.join(base_path, inventory), "rb") as f:
        for item in f.readlines():
            fn = item.strip()
            res.append(fn)
    pickle.dump(res, open(dump_path, "wb"))
    return res


def valid_files(base_path, path_list):
    black_list = {}
    for fn in path_list:
        if fn[:3] in black_list:
            continue
        if not os.path.exists(os.path.join(base_path, fn)):
            print(os.path.join(base_path, fn))
            black_list[fn[:3]] = 0
    return black_list


if __name__ == "__main__":

    print("loading...")
    bpa = "/data0/home/phonic/Code/c2f-vol-train/data/h36m/annot/"
    bpi = "/data0/home/phonic/Code/c2f-vol-train/data/h36m/images/"

    l1 = load_names(bpa, "valid_images.txt")
    l2 = load_names(bpa, "train_images.txt")
    print(len(l1))
    print(len(l2))
    print(valid_files(bpi, l1))
    print(valid_files(bpi, l2))
    print("passed")

