import yaml
import os

def save_yaml(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f, sort_keys=False)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def write_remidx(M, K, ppath, name) :
    outfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')
    f = open(outfile, 'w')
    s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M,K)]
    f.writelines(s)
    f.close()