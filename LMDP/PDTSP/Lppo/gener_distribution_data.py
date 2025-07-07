import os
import numpy as np
import pickle
from sklearn.cluster import KMeans
import random

# 生成单个实例的函数，现在接受一个分布类型参数
def generate_instance(distribution_type, problem_type,num_clusters=None, std_dev=None):
    problem_size = 101  # 假设我们仍然有21个点，但实际需求可能不同
    num_customers = 50  # 假设有10个取货点，对应10个送货点
    #instance = np.zeros((problem_size, 3))  # 现在第三维用于存储需求（正数为取货，负数为送货）
    instance = np.zeros((problem_size, 3))

    if distribution_type == 'uniform':
        # 均匀分布（作为基准或对比）
        instance[:, :2] = np.random.rand(problem_size, 2)
        # 生成需求（示例：每个取货点有随机需求，送货点需求相反）
        demands = np.random.randint(1, 11, size=num_customers)
        if problem_type == 'mptsp':
            instance[0,2] = 0
            instance[1:num_customers+1, 2] = demands
            instance[num_customers+1:, 2] = -demands  # 最后一个送货点需求与第一个取货点相反（简化处理）
        # print(instance)
        # 注意：这里简化了送货点需求的设置，实际中可能需要更复杂的逻辑

    elif distribution_type == 'clustered':
        # 聚类分布
        if num_clusters is None:
            raise ValueError("For clustered distribution, num_clusters must be specified.")
        # 生成聚类中心
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto',random_state=42).fit(np.random.rand(num_clusters * 2, 2))
        # 为每个点分配一个聚类，并根据聚类中心生成坐标
        labels = kmeans.predict(np.random.rand(problem_size, 2))
        instance[:, :2] = kmeans.cluster_centers_[labels] + np.random.randn(problem_size, 2) * 0.05  # 加上一点噪声
        # 生成需求（同上）
        if problem_type == 'mptsp':
            instance[0, 2] = 0
            demands = np.random.randint(1, 11, size=num_customers)
            instance[1:num_customers+1, 2] = demands
            instance[num_customers+1:, 2] = -demands  # 简化处理

    elif distribution_type == 'mixed':
        # 混合分布：这里我们简单地将点分为两部分，一部分均匀分布，一部分高斯分布
        half_size = problem_size // 2
        # 均匀分布部分
        instance[:half_size, :2] = np.random.rand(half_size, 2)
        # 高斯分布部分（假设以聚类中心为中心的高斯分布，但这里为了简化直接在整个空间内生成）
        gaussian_points = np.random.randn(half_size, 2) * std_dev + 0.5  # 缩放到0-1范围内并平移
        instance[half_size+1:, :2] = np.clip(gaussian_points, 0, 1)  # 确保坐标在0-1范围内
        # 生成需求（同上，但注意混合分布下可能需要更复杂的逻辑来处理取送货点的分配）
        if problem_type == 'mptsp':
            demands = np.random.randint(1, 11, size=num_customers)
            # 这里为了简化，我们仍然假设前num_customers个点为取货点
            instance[0, 2] = 0
            instance[1:num_customers+1, 2] = demands
            instance[num_customers+1:, 2] = -demands # 简化处理，最后一个送货点与第一个取货点需求相反

    elif distribution_type == 'gaussian':
        # 不同标准差的高斯分布
        if std_dev is None:
            raise ValueError("For gaussian distribution, std_dev must be specified.")
        # 生成高斯分布的坐标
        instance[:, :2] = np.random.randn(problem_size, 2) * std_dev + 0.5  # 缩放到0-1范围内并平移
        instance[:, :2] = np.clip(instance[:, :2], 0, 1)  # 确保坐标在0-1范围内
        # 生成需求（同上）\
        if problem_type == 'mptsp':
            demands = np.random.randint(1, 11, size=num_customers)
            instance[0, 2] = 0
            instance[1:num_customers+1, 2] = demands
            instance[num_customers+1:, 2] = -demands # 简化处理

    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

    # 注意：上述代码中的需求生成部分是为了演示而简化的。
    # 在实际应用中，您可能需要更复杂的逻辑来确保取送货点的匹配和需求的合理性。

    return instance

# 生成并保存多个实例的函数（现在接受分布类型和额外参数）
def generate_and_save_instances(file_path, problem_type,num_instances, distribution_type, **kwargs):
    instances = [generate_instance(distribution_type,problem_type, **kwargs) for _ in range(num_instances)]
    with open(file_path, 'wb') as f:
        pickle.dump(instances, f)

# 加载并访问特定实例的函数（不变）
def load_and_access_instances(file_path, instance_index):
    with open(file_path, 'rb') as f:
        instances = pickle.load(f)
    if 0 <= instance_index < len(instances):
        print(f"Total instances: {len(instances)}")
        return instances[instance_index]
    else:
        raise IndexError("Instance index out of range")

# 主程序（示例：生成不同分布的实例并保存）
if __name__ == "__main__":
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'mdatasets')
    os.makedirs(data_directory, exist_ok=True)
    problem_type = 'mpdtsp'
    # 生成并保存均匀分布的实例
    uniform_file_path = os.path.join(data_directory, 'uniform100_pdtsp.pkl')
    generate_and_save_instances(uniform_file_path, problem_type,num_instances=100, distribution_type='uniform')

    # 生成并保存聚类分布的实例（假设3个聚类）
    clustered_file_path = os.path.join(data_directory, 'clustered100_pdtsp.pkl')
    generate_and_save_instances(clustered_file_path, problem_type,num_instances=100, distribution_type='clustered', num_clusters=3)

    # 生成并保存混合分布的实例（高斯部分标准差为0.2）
    mixed_file_path = os.path.join(data_directory, 'mixed100_pdtsp.pkl')
    generate_and_save_instances(mixed_file_path, problem_type,num_instances=100, distribution_type='mixed', std_dev=0.2)

    # 生成并保存不同标准差的高斯分布实例（标准差为0.5）
    gaussian_file_path = os.path.join(data_directory, 'gaussian_100_pdtsp_0.5.pkl')
    generate_and_save_instances(gaussian_file_path,problem_type, num_instances=100, distribution_type='gaussian', std_dev=0.5)

    gaussian_file_path = os.path.join(data_directory, 'gaussian_100_pdtsp_1.pkl')
    generate_and_save_instances(gaussian_file_path, problem_type, num_instances=100, distribution_type='gaussian',
                                std_dev=1)

    # 加载并访问特定实例（示例）
    instance_index = 0
    # print("Loading uniform distribution instance:")
    # uniform_instance = load_and_access_instances(uniform_file_path, instance_index)
    # print(uniform_instance)