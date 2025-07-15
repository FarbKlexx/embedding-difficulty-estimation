# embedding-difficulty-estimation
This repository contains the source code for my (Adrian Uhe) bachelor thesis.
## Setup
> [!NOTE]
> To correctly pull all the files it is recommended to install git-lfs. More information about that on https://git-lfs.com/.

After cloning the repository (and I recommend to create a new environment), to install all required dependencies execute:
```
pip install -r requirements.txt
```
If you want to calculate the embeddings on your GPU (if available) you can additionally install the following dependencies:
> [!WARNING]
> These dependencies can be quite large (~3GB) and only work with CUDA compatible NVIDIA Graphic Cards.

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Code Structure
## Libraries

# markdown-Syntax
## H2
### H3

---

**bold**
*cursive*
`inline code`

```python 
def hello(): 
  print("Hello, Markdown!")
```

- Punkt A
- Punkt B

1. Erster Schritt
2. Zweiter Schritt

| Datei            | Beschreibung              |
|------------------|---------------------------|
| `main.py`        | Startpunkt des Programms  |
| `model.pkl`      | Trainiertes ML-Modell     |

[Zum Abstract](#abstract)

> [!NOTE]
> Text

> [!TIP]
> Text



> [!IMPORTANT]
> Text

> [!CAUTION]
> Text
