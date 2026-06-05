from gpatch_v4.utils import safe_import_class


class CustomActorRegistry:
    """Registry for user-defined custom actor classes.

    Allows training configurations to reference custom actor
    implementations by name.
    """

    registered_actor_class = {}

    @classmethod
    def register(cls, name, actor_class):
        """Register a custom actor class under the given name.

        Parameters
        ----------
        name : str
        actor_class : type
        """
        assert name not in cls.registered_actor_class
        cls.registered_actor_class[name] = actor_class

    @classmethod
    def get(cls, name):
        """Retrieve a registered actor class by name, or *None* if absent.

        Parameters
        ----------
        name : str

        Returns
        -------
        type or None
        """
        return cls.registered_actor_class.get(name, None)


def register_custom_actors(rl_training_config):
    """Import and register all custom actors declared in the config.

    Parameters
    ----------
    rl_training_config : object
        Training config with a ``custom_actors`` list of ``(cls_path, name)``.
    """
    if not rl_training_config.custom_actors:
        return
    for e in rl_training_config.custom_actors:
        cls = safe_import_class(e.cls_path)
        CustomActorRegistry.register(e.name, cls)
