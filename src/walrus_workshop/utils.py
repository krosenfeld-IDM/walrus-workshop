import inspect


def filter_kwargs(kwargs, module):
    """
    Filter a dictionary of kwargs to only include keys that the module accepts.
    """

    # Get the list of arguments SAE expects
    module_args = set(inspect.signature(module.__init__).parameters)

    # Create a filtered dict containing ONLY keys that SAE accepts
    model_kwargs = {k: v for k, v in kwargs.items() if k in module_args}
    return model_kwargs
