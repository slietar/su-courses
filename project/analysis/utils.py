import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional, ParamSpec, TypeVar


cache_path = Path.home() / '.cache' / 'su-project'
cache_path.mkdir(exist_ok=True, parents=True)


P = ParamSpec('P')
T = TypeVar('T')

def cache(name: Optional[str] = None):
  def decorator(fn: Callable[P, T], /):
    def new_fn(*args: P.args, **kwargs: P.kwargs):
      if (name is None) and (fn.__module__ == '__main__'):
        return fn(*args, **kwargs)

      cache_name = f'{fn.__module__}:{fn.__qualname__}' if name is None else name
      path = cache_path / cache_name

      if path.exists():
        with path.open('rb') as file:
          result: T = pickle.load(file)

        return result
      else:
        result = fn(*args, **kwargs)

        with NamedTemporaryFile('wb', delete=False) as file:
          pickle.dump(result, file)

        Path(file.name).replace(path)

        return result

    return new_fn

  return decorator


if __name__ == '__main__':
  import shutil

  print('Clearing cache')
  shutil.rmtree(cache_path)
