# Practical challenge 

## Project overview:
This program takes images as input, detects a specific object within them, segments the object, and extracts its dominant colors.

## Directory structure
```
.
├── data/               # Folder for input images.
│   ├── image1.jpg
│   └── image2.png      
├── model/
│   └── yolov8n-seg.pt  # The script downloads this if not present.
├── Figures/            # Output is saved here.
├── config.yaml         # For configuring paths.
├── main.py             # Main Python file.
├── docs                # Folder for research documents. 
├── .gitignore          
└── README.md

```


## Setup & installation

1. Clone repository

    ```
    git clone <repo>
    ```

2. Create virtual environment and install `requirements.txt`

    ```
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage
1. Add images into data folder.
3. Run Python script

    ```
    python main.py
    ```

