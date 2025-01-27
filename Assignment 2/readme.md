

# Data Analytics Project
## Link to the Colab Notebook

- [Colab Notebook](https://colab.research.google.com/drive/1dOBYhnXqwnwuQ9mLqwBDCZW8Lhfopuai?usp=sharing)

## Description

This project includes a script to perform data analytics, with all dependencies listed in `requirements.txt`.

## Setup Instructions

### Prerequisites

- Python 3.x should be installed on your system.

### Install Dependencies

1. **Create a Virtual Environment and Install Dependencies**

   ```bash
   make all
   ```

   This command will:
   - Create a virtual environment named `venv`.
   - Install all required dependencies listed in `requirements.txt`.
   - Run the `Solution.py` script.

### Running the Code

To run `Solution.py` after setting up the environment, use:

```bash
make run
```

This will ensure that dependencies are installed and the script is executed.

### Cleaning Up

To remove all `.png` files generated by the script, use:

```bash
make clean
```

### Removing the Virtual Environment

If you want to delete the virtual environment, use:

```bash
make clean-env
```
