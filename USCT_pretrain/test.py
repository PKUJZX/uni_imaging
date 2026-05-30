import h5py

def print_h5_structure(name, obj):
    print(name)
    # 如果是数据集，打印其形状
    if isinstance(obj, h5py.Dataset):
        print('Shape:', obj.shape)
        print('Type:', obj.dtype)

with h5py.File('try_train.h5', 'r') as f:
    # 打印文件结构
    f.visititems(print_h5_structure)