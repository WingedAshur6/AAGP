#%%
import os, venv, subprocess, platform

# [1] - grab the directories
host_folder=os.path.dirname(os.path.abspath(__file__))
venv_folder= f'{host_folder}/venv'
try:
    os.mkdir(venv_folder)
except:
    pass
print('\n'*2)
print(f'Current directory: {host_folder}')
print(f'Venv directory   : {venv_folder}')

# [2] - make a venv
vb = venv.EnvBuilder(
    system_site_packages=False,
    clear=True,
    symlinks=False,
    with_pip=True
)
vb.create(env_dir=venv_folder)

# [3] - set the desired file
exe_script = f'"{host_folder}/EXECUTOR.py"'
print(f'Running          : {exe_script}')

# [4] - Determine activation command based on OS
if platform.system() == 'Windows':
    activate_cmd = os.path.join(venv_folder, 'Scripts', 'activate')
else:
    activate_cmd = os.path.join(venv_folder, 'bin', 'activate')
activate_cmd = f'"{activate_cmd}"'

# [5] - Combine activation with running the script
print(f'Activate command : {activate_cmd}')
if platform.system() == 'Windows':
    command = f"{activate_cmd} && python {exe_script}"
    subprocess.run(command, shell=True)
else:
    command = f"source {activate_cmd} && python {exe_script}"
    subprocess.run(command, shell=True, executable="/bin/bash")