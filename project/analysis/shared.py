from pathlib import Path


root_path = Path(__file__).parent / '..'
output_path = root_path / 'output'

output_path.mkdir(exist_ok=True, parents=True)
