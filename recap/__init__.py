"""RECAP v8.3; load the training stack only when its public classes are used."""
__all__ = ["DAProDeconvConfig", "DAProDeconvPipeline"]


def __getattr__(name):
    if name in __all__:
        from .trainer import DAProDeconvConfig, DAProDeconvPipeline
        globals().update(DAProDeconvConfig=DAProDeconvConfig, DAProDeconvPipeline=DAProDeconvPipeline)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
