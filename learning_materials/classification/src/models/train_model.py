import sklearn


def instantiate_model(model, model_config):
    """
    initiate model using eval, implement with defensive programming
    Args:
        ensemble_model [str]: name of the ensemble model

    Returns:
        [sklearn.model]: initiated model
    """
    if model in dir(sklearn.neighbors):
        return eval("sklearn.neighbors." + model)(**model_config)
    if model in dir(sklearn.ensemble):
        return eval("sklearn.ensemble." + model)(**model_config)
    if model in dir(sklearn.tree):
        return eval("sklearn.tree." + model)(**model_config)
    else:
        raise NameError(f"{model} is not in sklearn.")
