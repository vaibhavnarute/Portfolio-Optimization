import os
import sys

def create_directory_structure():
    """
    Create the project directory structure.
    """
    # Define directories to create
    directories = [
        "api",
        "models",
        "utils",
        "data",
        "tests"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py in each directory
        with open(os.path.join(directory, "__init__.py"), "w") as f:
            f.write("# This file makes the {} directory a Python package\n".format(directory))
    
    print("Directory structure created successfully!")

def create_empty_files():
    """
    Create empty files for the project.
    """
    # Define files to create
    files = [
        "api/main.py",
        "models/markowitz.py",
        "models/monte_carlo.py",
        "models/factor_models.py",
        "models/reinforcement.py",
        "utils/data_loader.py",
        "config.py",
        "main.py"
    ]
    
    # Create files
    for file in files:
        with open(file, "w") as f:
            f.write("# TODO: Implement {}\n".format(file))
    
    print("Empty files created successfully!")

def main():
    """
    Main function to set up the project.
    """
    print("Setting up portfolio optimization project...")
    
    # Create directory structure
    create_directory_structure()
    
    # Create empty files
    create_empty_files()
    
    print("Project setup complete!")
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Implement the models and utilities")
    print("3. Run the project: python main.py --mode markowitz")

if __name__ == "__main__":
    main()