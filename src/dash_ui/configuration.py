"""Forms describe only parameters consumed by the existing optimizers."""
from src.models.registry import ModelRegistry

ALGORITHMS = {
    '两阶段算法(推荐)': [],
    'PSO': [('n_particles', '粒子数', 100, 10, 100, 1), ('w', '惯性权重', .7, 0, 1, .01),
            ('c1', '个体学习因子', 1.5, 0, 2, .1), ('c2', '群体学习因子', 1.5, 0, 2, .1)],
    'SCE-UA': [],
    '差分进化(DE)': [('mutation_factor', '变异因子', .8, .01, 1.99, .01),
                     ('crossover_prob', '交叉概率', .7, 0, 1, .01),
                     ('pop_size', '种群倍数（每参数维度）', 50, 5, 100, 1)],
    '遗传算法(GA)': [('pop_size', '种群大小', 50, 10, 100, 1),
                     ('crossover_rate', '交叉率', .8, 0, 1, .01),
                     ('mutation_rate', '变异率', .1, 0, 1, .01)],
}
FIELDS = ['mode', 'area', 'timestep', 'warmup_hours', 'parameter_mode', 'algorithm', 'max_iter',
          'models', 'routing', 'routing_fixed', 'routing-k', 'routing-x',
          'xgb_depth', 'xgb_rate', 'xgb_estimators', 'xgb_seed', 'xgb_quantile', 'xgb_ratio', 'xgb_clip']
FIELDS += ['col-' + name for name in ('date', 'precip', 'evap', 'flow', 'upstream')]
PARAM_IDS = [f'algo-{i}-{name}' for i, fields in enumerate(ALGORITHMS.values()) for name, *_ in fields]
CONFIG_IDS = FIELDS + PARAM_IDS
DATA_KEYS = ('mode', 'area', 'timestep', 'mapping', 'routing', 'warmup_hours')
SIMULATION_KEYS = ('warmup_hours', 'parameter_mode', 'algorithm', 'max_iter', 'models',
                   'algo_params', 'routing_fixed', 'routing_parameters')
POST_KEYS = ('xgb_depth', 'xgb_rate', 'xgb_estimators', 'xgb_seed', 'xgb_quantile', 'xgb_ratio', 'xgb_clip')


def parse_config(values):
    """Validate user-controlled numerical fields once, when accepting a form."""
    data = dict(zip(CONFIG_IDS, values))
    config = {key: data[key] for key in FIELDS if not key.startswith(('col-', 'routing-'))}
    for key, lower, upper, integer in [('area', .1, 100000, False), ('warmup_hours', 0, 720, True),
                                      ('max_iter', 5, 200, True), ('xgb_depth', 3, 15, True),
                                      ('xgb_rate', .01, .5, False), ('xgb_estimators', 50, 500, True),
                                      ('xgb_seed', 1, 9999, True), ('xgb_quantile', .8, .98, False),
                                      ('xgb_ratio', .1, 1, False)]:
        value = data[key]
        if value is None or not lower <= value <= upper or (integer and int(value) != value):
            raise ValueError(f'{key} 须在 {lower}–{upper} 之间' + ('且为整数。' if integer else '。'))
    for key in ('routing', 'routing_fixed', 'xgb_clip'):
        config[key] = bool(data[key])
    k, x = data['routing-k'], data['routing-x']
    if k is None or x is None or not .5 <= k <= 10 or not 0 <= x <= .5:
        raise ValueError('汇流参数须满足 k=0.5–10、x=0–0.5。')
    config['routing_parameters'] = {'k_routing': k, 'x_routing': x}
    config['mapping'] = {key: data['col-' + key] for key in ('date','precip','evap','flow','upstream')}
    if not config['models']:
        raise ValueError('请至少选择一个水文模型。')
    for name in config['models']:
        if config['timestep'] == 'hourly' and not ModelRegistry.get_model(name).supports_hourly:
            raise ValueError(f'{name} 仅支持日尺度，请取消选择或修改时间尺度。')
    config['algo_params'] = {}
    index = list(ALGORITHMS).index(config['algorithm'])
    for name, label, default, lower, upper, step in ALGORITHMS[config['algorithm']]:
        value = data[f'algo-{index}-{name}']
        if value is None or not lower <= value <= upper or (step == 1 and int(value) != value):
            raise ValueError(f'{label} 须在 {lower}–{upper} 之间' + ('且为整数。' if step == 1 else '。'))
        config['algo_params'][name] = value
    return config
