import ast
import subprocess
import sys

def get_imports(filename):
    with open(filename, 'r') as file:
        tree = ast.parse(file.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                imports.add(f"{node.module}.{n.name}")

    return imports

def install_dependencies(filename):
    imports = get_imports(filename)
    installed_modules = {line.split('==')[0] for line in subprocess.check_output(['pip', 'freeze']).decode().strip().split('\n')}
    
    for module in imports:
        if module not in installed_modules:
            try:
                subprocess.call([sys.executable, 'm', 'pip', 'install', module])
            except subprocess.CalledProcessError as e:
                print(f"Error installing {module}: {e}")
                raise ValueError(f"Error installing {module}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python install_dependencies.py <your_script.py>")
        sys.exit(1)

    python_script = sys.argv[1]
    install_dependencies(python_script)
