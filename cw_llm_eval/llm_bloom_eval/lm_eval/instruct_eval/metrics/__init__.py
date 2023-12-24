def get_metric(metric_type, num_player):
    if metric_type == "score":
        from .score import ScoreEval as METRIC
        return METRIC(num_player)
    elif metric_type == "rank":
        from .rank import RankEval as METRIC
        return METRIC()
    elif metric_type == "rank_old":
        from .rank import RankEval as METRIC
        return METRIC()
    elif metric_type == "score_pair":
        from .score_pair import ScorePairEval as METRIC
        return METRIC()
    else:
        raise RuntimeError('Unknown metric_type: %s'% metric_type)