import os

import numpy as np
import pickle


# 生成单个实例的函数
def generate_instance():
    problem_size = 21
    costumer = 10
    # 初始化一个形状为(51, 2, 3)的numpy数组，用于存储点的信息
    instance = np.zeros((problem_size, 2))

    # 生成坐标（前两维）
    instance[:, :2] = np.random.rand(problem_size, 2)  # 在0-1之间随机生成坐标

    # 生成需求（最后一维）
    # demands = np.random.randint(1, 11, size=costumer)  # 为取货点生成1到10之间的随机整数需求
    # instance[0, 2] = 20
    # instance[1:costumer+1,  2] = demands  # 取货点的需求为正
    # instance[costumer+1:, 2] = -demands  # 送货点的需求为负，与取货点对应

    # 确保第i个点与第i+25个点的需求维为相反数（这一步其实是多余的，因为上面已经直接设置了）
    # 但为了清晰起见，还是保留了这个检查
    # assert np.all(instance[1:costumer+1, 2] == -instance[costumer+1:, 2])

    # 由于我们已经为前26个取货点和对应的25个送货点设置了需求，
    # 最后一个送货点（即第51个点）的需求是自动满足相反数关系的第26个取货点的相反数，
    # 但为了完整性，我们再次确认这一点（实际上在上面的设置中已经隐含了这个关系）
    # 注意：这里我们并没有显式地为第51个点设置需求，因为它应该自动与第26个点相反
    # 但为了代码清晰，我们可以添加一个断言来检查这一点（虽然它是冗余的）
    # assert instance[25, :, 2] == -instance[50, :, 2]  # 这行是冗余的，因为已经通过上面的设置隐含了

    # 不过，为了完整性，我们可以直接设置第51个点的需求（虽然这是多余的）
    # instance[50, :, 2] = -instance[25, :, 2]  # 这行是多余的，不需要执行

    # 返回生成的实例
    # print(instance)
    return instance


# 生成1000个实例并保存到pickle文件中
def generate_and_save_instances(file_path, num_instances=2000):
    instances = [generate_instance() for _ in range(num_instances)]
    with open(file_path, 'wb') as f:
        pickle.dump(instances, f)

    # 从pickle文件中加载实例并访问它们


def load_and_access_instances(file_path, instance_index):
    with open(file_path, 'rb') as f:
        instances = pickle.load(f)
    if 0 <= instance_index < len(instances):
        print("instances",len(instances))
        return instances[instance_index]
    else:
        raise IndexError("Instance index out of range")

    # 使用示例


if __name__ == "__main__":
    # 生成并保存实例到文件
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'datasets')
    # file_path = './datasets/data1.pkl'
    pkl_file_path = os.path.join(data_directory, 'data20_2000.pkl')
    generate_and_save_instances(pkl_file_path)

    # 加载并访问特定实例
    instance_index = 1  # 例如，访问第6个实例（索引从0开始）
    instance = load_and_access_instances(pkl_file_path, instance_index)
    print(instance)