import h5py

file = h5py.File('weights.h5', 'r')

keys = list(file.keys())
print('Top-level datasets and groups:')
for key in keys:
    print(key)

print('All datasets and groups:')
def print_name(name):
    print(name)

file.visit(print_name)


file.close()