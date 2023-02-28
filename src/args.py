import argparse


class BaseArgsAlgo():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--algo', type=str, default='sac')
        parser.add_argument('--lr', type=float, default=0.00001)
        parser.add_argument('--num_episodes', type=int, default=4000)
        parser.add_argument('--num_iters', type=int, default=2000)
        parser.add_argument('--max_ep_len', type=int, default=200)
        parser.add_argument('--num_test_episodes', type=int, default=4)
        parser.add_argument('--render', action='store_true')
        parser.add_argument('--sess', type=str, default='test')
        parser.add_argument('--batch_size', type=int, default=100, help='num of transitions sampled from replay buffer')
        parser.add_argument('--replay_size', type=int, default=int(2e4), help='max size of the replay buffer')
        parser.add_argument('--save_freq', type=int, default=50, help='frequency for saving the model')
        parser.add_argument('--plot_figs', action='store_true', default=False)
        parser.add_argument('--log_losses', action='store_true', default=False)
        parser.add_argument('--has_checkpoint', action='store_true', default=False)
        parser.add_argument('--epoch_checkpoint', type=int, default=20000)

        # Soft Actor-Critic params
        parser.add_argument('--gamma', type=float, default=0.99, help='discount for future rewards')
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--policy_noise', type=float, default=0.2, help='target policy smoothing noise')
        parser.add_argument('--exploration_noise', type=float, default=0.1)
        parser.add_argument('--polyak', type=float, default=0.995, help='target policy update parameter (1-tau)')
        parser.add_argument('--noise_clip', type=float, default=0.5)
        parser.add_argument('--policy_delay', type=float, default=2, help='delayed policy updates parameter')

        return parser

    def get_args(self):
        parser = argparse.ArgumentParser()
        return self.initialize(parser).parse_args()


class BaseArgsEnv(BaseArgsAlgo):
    def initialize(self, parser):
        parser = BaseArgsAlgo.initialize(self, parser)
        parser.add_argument('--env_name', type=str, default='Track2d-v1')
        parser.add_argument('--len_workspace', type=int, default=20)
        parser.add_argument('--n_actions', type=int, default=2)
        parser.add_argument('--n_drones', type=int, default=1)
        parser.add_argument('--safe_rad', type=float, default=6.0)
        parser.add_argument('--pos_noise', type=float, default=0.5)
        parser.add_argument('--ori_noise', type=float, default=1)
        parser.add_argument('--step_size_r', type=float, default=1)
        parser.add_argument('--step_size_t', type=float, default=0.2)
        parser.add_argument('--reward_dist', type=float, default=7.5)
        parser.add_argument('--reward_k', type=float, default=2.0)
        parser.add_argument('--reward_punish', type=float, default=-10)

        # Args for multi-actor tracking envs.
        parser.add_argument('--num_actors', type=int, default=2)
        parser.add_argument('--vp_workspace', type=int, default=60)
        parser.add_argument('--view_range', type=int, default=30)
        parser.add_argument('--step_size_r_d', type=float, default=5)
        parser.add_argument('--step_size_r_a', type=float, default=1)

        # Args for Dynamic Programming.
        parser.add_argument('--T', type=int, default=10)
        parser.add_argument('--create_cmap', type=bool, default=False)
        return parser


class BaseArgs(BaseArgsEnv):
    def initialize(self, parser):
        parser = BaseArgsEnv.initialize(self, parser)
        parser.add_argument('--seed', type=int, default=0)
        return parser


class TestArgs(BaseArgsEnv):
    def initialize(self, parser):
        parser = BaseArgsEnv.initialize(self, parser)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--num_test_eps', type=int, default=100)
        parser.add_argument('--render_every', type=int, default=1)
        parser.add_argument('--fname', type=str, default='exp')
        parser.add_argument('--post_name', type=str, default='tspn')
        return parser


class VPArgs:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--env_name', type=str, default='view_planning')
        parser.add_argument('--len_workspace', type=float, default=20.0)
        parser.add_argument('--safe_rad', type=float, default=6.0)
        parser.add_argument('--step_size_t', type=float, default=0.5)
        parser.add_argument('--render', type=bool, default=True)
        parser.add_argument('--ppa_thr', type=float, default=0.12)
        parser.add_argument('--n_drones', type=int, default=3)
        parser.add_argument('--n_actors', type=int, default=1)
        parser.add_argument('--time_sleep', type=float, default=0.05)
        parser.add_argument('--scale', type=float, default=1.0)
        parser.add_argument('--obj_name', type=str, default='person_actor')
        parser.add_argument('--post_name', type=str, default='tspn')
        parser.add_argument('--fname', type=str, default='exp_wafr')
        return parser

    def get_args(self):
        parser = argparse.ArgumentParser()
        return self.initialize(parser).parse_args()
