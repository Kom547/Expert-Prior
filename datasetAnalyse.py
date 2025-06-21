from expert_imitation_learning_MoE import calculate_action_proportions,load_expert_samples
from config import get_config
parser = get_config()
# 读取参数

args = parser.parse_args()
#加载处理数据
OBS, ACT = load_expert_samples(args.expert_data_path)
calculate_action_proportions(ACT)