import pickle

# 加载 .pkl 文件
import numpy

with open('E:\python_project\my-LMDP\PDTSP\lppoNet\datasets\pdp_20.pkl', 'rb') as f:
    data = pickle.load(f)

# 检查数据是否可以切片（这里假设 data 是一个列表或 NumPy 数组）
if isinstance(data, (list, numpy.ndarray)):
    # 获取前 500 个数据
    first_500_data = data[:500]

    # 计算要替换的数据范围（从索引 500 到 1999，共 1500 个元素）
    # 注意：Python 索引是从 0 开始的，所以 2000 的索引实际上是第 2001 个元素（如果不存在的话）
    # 因此，我们要替换到索引 1999（即第 2000 个元素之前的最后一个元素）
    replacement_length = 1500

    # 使用前 500 个数据重复来填充 501 到 2000 的位置
    # 这里简单地重复前 500 个数据，但你也可以根据需要生成其他数据
    # 注意：这种重复可能不是你所期望的，这里只是为了演示如何替换数据
    # replacement_data = first_500_data * (replacement_length // len(first_500_data) + 1)[:replacement_length]
    # replacement_data = replacement_data[:replacement_length]  # 确保数据长度正确（尽管上面的乘法已经考虑了这一点）

    # 替换数据
    data[499:999] = first_500_data
    data[999:1499]=first_500_data
    data[1499:1999]=first_500_data
else:
    raise ValueError("Loaded data is not a list or NumPy array and cannot be sliced in this way.")

# 保存修改后的数据回文件（可以覆盖原文件或保存为新文件）
with open('E:\python_project\my-LMDP\PDTSP\lppoNet\datasets\pdp_20_1.pkl', 'wb') as f:
    pickle.dump(data, f)

# 如果你想要覆盖原文件，可以将上面的保存路径改回原文件的路径
# with open('path_to_your_file.pkl', 'wb') as f:
#     pickle.dump(data, f)