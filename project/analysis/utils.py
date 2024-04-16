from pathlib import Path
import pickle
import shutil
from tempfile import NamedTemporaryFile
from typing import Callable, ParamSpec, TypeVar


cache_path = Path.home() / '.cache' / 'su-project'
cache_path.mkdir(exist_ok=True, parents=True)


P = ParamSpec('P')
T = TypeVar('T')

def cache(fn: Callable[P, T], /):
  def new_fn(*args: P.args, **kwargs: P.kwargs):
    name = f'{fn.__module__}:{fn.__qualname__}'
    path = cache_path / name

    if fn.__module__ == '__main__':
      path.unlink(missing_ok=True)
      return fn(*args, **kwargs)

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


if __name__ == '__main__':
  print('Clearing cache')
  shutil.rmtree(cache_path)
