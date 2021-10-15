import subprocess
from pathlib import Path

for script in Path('.').glob('draw_binned_Rr_profiles_*.py'):
    subprocess.run(['python', str(script)])
