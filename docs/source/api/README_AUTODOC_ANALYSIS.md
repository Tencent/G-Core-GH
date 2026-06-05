# GPatch V4 Autodoc API Documentation - Implementation Guide

This directory contains analysis and sample files for adding autodoc API documentation to the GPatch V4 framework.

## Files in This Analysis

1. **GPATCH_V4_MODULE_ANALYSIS.md** (root directory)
   - Comprehensive module-by-module analysis
   - Detailed docstring assessment for each of 19 top-level modules
   - Status indicators (✅ Ready, ⚠️ Partial, ❌ Not Recommended)
   - Implementation recommendations

2. **QUICK_REFERENCE.txt** (root directory)
   - Quick lookup table for module status
   - Implementation checklist with 5 phases
   - Docstring style guidelines
   - Example Sphinx directives

3. **Sample RST Files** (in this directory, prefixed with SAMPLE_)
   - SAMPLE_core.rst - Core module template
   - SAMPLE_utils.rst - Utils module template
   - SAMPLE_training_backend.rst - Training backend template

## Key Findings

### Ready for Autodoc (13 modules)
Modules with `__init__.py` and extensive docstrings:
- ✅ core, utils, training_backend, generation_backend (Priority 1)
- ✅ orches, rollout_generator, rpc_client (Priority 2)
- ✅ trainer, reward, actor (Priority 3)
- ✅ client, extended_model, extended_pipeline (Priority 4)

### Partial (need __init__.py first)
- ⚠️ configs - Create __init__.py exporting key config classes
- ⚠️ rollout - Create __init__.py for RolloutManager, RolloutCoordinator
- ⚠️ agentic - Create __init__.py if prioritizing agentic features

### Not Recommended
- ❌ entry - CLI scripts (document in tutorials)
- ❌ models - Internal architecture registry
- ❌ patch - Unknown/minimal content
- ❌ evaluate - Minimal public API docstrings

## Implementation Phases

### Phase 1: Foundation (High Priority)
Create these 4 RST files for core infrastructure:
```
core.rst                 - Advantage computation, registries
utils.rst                - Logging, communication, data utils
training_backend.rst     - Training engine abstractions
generation_backend.rst   - Inference engine abstractions
```

### Phase 2: Orchestration
Create these 3 RST files:
```
orches.rst              - Ray orchestration
rollout_generator.rst   - Rollout management
rpc_client.rst          - RPC communication
```

### Phase 3: Training & Rewards
Create/update these 3 RST files:
```
trainer.rst             - Update existing trainer_v4.rst
reward.rst              - Reward computation (or update trainer_v4.rst)
actor.rst               - Ray actor implementations
```

### Phase 4: Specialized
Create these 3 RST files:
```
client.rst              - RPC clients for engines
extended_model.rst      - Model-specific implementations
extended_pipeline.rst   - Model-specific pipelines
```

### Phase 5: Infrastructure (Optional)
After creating __init__.py files:
```
Create configs/__init__.py → configs.rst
Create rollout/__init__.py → rollout.rst
Create agentic/__init__.py → agentic.rst (if prioritized)
```

## Using the Sample Files

The SAMPLE_*.rst files show the recommended structure:

1. **Section Organization**
   - Group related classes/functions under descriptive subsections
   - Use clear hierarchy with `---`, `^^^` for RST headers

2. **Autodoc Directives**
   - Use `.. autoclass::` for classes with `:members:` and `:undoc-members:`
   - Use `.. automodule::` for specific function exports
   - Use `:noindex:` when showing module-level exports

3. **Docstring Style**
   - All modules use NumPy-style docstrings
   - Supported sections: Parameters, Returns, Raises, Notes, Examples
   - Factory classes highlight return types clearly

## Common Patterns Found

### 1. Factory Classes
Most modules have factory classes with documented methods:
```python
class SomeFactory:
    @staticmethod
    def get_something(config, **kwargs):
        """Create appropriate instance based on config.
        
        Parameters
        ----------
        config : object
            Configuration object
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        ConcreteClass
            Instantiated object
        """
        ...
```

**Documentation approach:**
```rst
.. autoclass:: gpatch_v4.module.SomeFactory
   :members: get_something
   :undoc-members:
```

### 2. Abstract Base Classes
Several modules define ABC patterns:
```python
class BaseAbc(ABC):
    """Abstract base class for X."""
    
    @abstractmethod
    def do_something(self):
        """Do something."""
        ...
```

**Documentation approach:**
```rst
.. autoclass:: gpatch_v4.module.BaseAbc
   :members:
   :undoc-members:
```

### 3. Mixins
Some modules use mixin classes for shared functionality:
```python
class SomeMixin:
    """Provides X functionality."""
    
    def method(self):
        """Documented method."""
        ...
```

**Documentation approach:**
```rst
.. autoclass:: gpatch_v4.module.SomeMixin
   :members:
   :undoc-members:
```

### 4. Module-level Registries
Several modules export registries and helper functions:
```python
# In __init__.py
BUILDIN_TYPES = list(REGISTRY.keys())

def register_custom_type(name, ...):
    """Register custom type."""
    ...
```

**Documentation approach:**
```rst
.. automodule:: gpatch_v4.module
   :members: register_custom_type, BUILDIN_TYPES
   :noindex:
```

## Docstring Examples

### Good Docstring (found in codebase)
```python
class TrainerRetryMixin:
    """Mixin providing retry logic for trainer launch."""
    
    async def launch_with_retry(self, config: RlConfig):
        """Launch training with automatic retries on failure.
        
        Parameters
        ----------
        config : RlConfig
            RL training configuration.
        
        Raises
        ------
        RuntimeError
            If all retry attempts fail.
        """
        ...
```

### Good Factory Docstring (found in codebase)
```python
class RolloutGeneratorFactory:
    """Factory that returns the appropriate rollout generator."""
    
    @staticmethod
    def get_rollout_generator(
        config,
        sampler_client,
        gen_rm_client,
        bt_rm_client,
        run_eval=False,
        **kwargs
    ) -> BaseRolloutGenerator:
        """Instantiate a rollout generator based on ``config.policy.rollout_gen_type``.
        
        Parameters
        ----------
        config : object
            Training configuration.
        sampler_client : object
            Sampler client.
        gen_rm_client : object
            Generative RM client.
        bt_rm_client : object
            BT RM client.
        run_eval : bool, optional
            If *True*, run in evaluation mode, by default *False*.
        **kwargs : dict
            Extra arguments forwarded to the generator constructor.
        
        Returns
        -------
        BaseRolloutGenerator
            Concrete rollout generator.
        """
        ...
```

## Checklist for Creating RST Files

For each new module RST file:

- [ ] Create file at `docs/source/api/{module}.rst`
- [ ] Add title with module path (e.g., "Core Module (``gpatch_v4.core``)")
- [ ] Add brief description of module purpose
- [ ] Group related classes/functions into logical sections
- [ ] Use `.. autoclass::` for each key class with `:members:` and `:undoc-members:`
- [ ] Use `.. automodule::` for functions with specific `:members:` list
- [ ] Use `:noindex:` for module-level exports to avoid duplicate targets
- [ ] Follow NumPy docstring conventions for formatting
- [ ] Include section headers for logical grouping (e.g., "Base Classes", "Implementations", etc.)

## Integration with Sphinx

These RST files are meant to be included in your Sphinx `index.rst` or main API reference document:

```rst
.. toctree::
   :caption: API Reference
   :hidden:
   
   api/core
   api/utils
   api/training_backend
   api/generation_backend
   api/orches
   api/rollout_generator
   api/rpc_client
   api/trainer
   api/reward
   api/actor
   api/client
   api/extended_model
   api/extended_pipeline
```

## Updating Existing Files

Existing RST files that should be reviewed/updated:

1. **trainer_v4.rst** - Add references to reward.rst or consolidate reward docs
2. **configs_v4.rst** - Already comprehensive, ensure it references any new configs module
3. Consider creating an index page that lists all API modules

## Tips for Success

1. **Start with Phase 1** - These are foundational and dependencies for higher phases
2. **Use samples as templates** - Copy SAMPLE_*.rst and adapt for each module
3. **Test with Sphinx** - Run `sphinx-build` to verify autodoc directives resolve
4. **Group logically** - Organize classes/functions by functionality, not file structure
5. **Keep it simple** - Don't over-document; let docstrings speak for themselves
6. **Review existing** - Look at trainer_v4.rst and configs_v4.rst for proven patterns

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Missing `__init__.py` for configs/rollout/agentic | Create `__init__.py` exporting key classes first |
| Autodoc can't find class | Verify full path matches actual import path |
| Duplicate reference errors | Use `:noindex:` on module-level exports |
| Docstrings not showing | Check NumPy format; ensure `"""` docstring syntax |
| Too much output | Use `:members:` with specific names instead of all members |

## Questions & Support

Refer to:
- GPATCH_V4_MODULE_ANALYSIS.md for detailed module assessment
- QUICK_REFERENCE.txt for implementation checklist
- SAMPLE_*.rst files for structure templates
- Sphinx autodoc documentation: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
