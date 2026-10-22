# proy-redesneuronales

Nutri-Score con redes neuronales (Trabajo 1, Grupo 1: Alejandro, Miguel y Javier).

Entrenamos una red neuronal en PyTorch que predice la letra del Nutri-Score (A a E) de un producto a partir de sus valores nutricionales por 100 g, con datos de Open Food Facts. El informe es una web hecha con Quarto.

## Instalación (la primera vez)

### 1. Programas necesarios

- [Python 3.11 o superior](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [VS Code](https://code.visualstudio.com/)
- [Quarto CLI](https://quarto.org/docs/get-started/)

Comprueba que responden:

```bash
python3 --version
git --version
quarto check
```

### 2. Clonar el repositorio y crear el entorno

```bash
git clone https://github.com/JavierCagigal/proy-redesneuronales.git
cd proy-redesneuronales
python3 -m venv .venv
```

Activa el entorno:

```bash
source .venv/bin/activate        # macOS y Linux
.venv\Scripts\activate           # Windows
```

Instala las librerías:

```bash
pip install -r requirements.txt
```

En Linux, PyTorch descarga por defecto la versión con GPU (varios GB). Para la de CPU, ejecuta esto **antes** del `pip install -r`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. VS Code

Abre la carpeta del repo en VS Code y acepta instalar las extensiones recomendadas (Python, Jupyter y Quarto). Cuando te pida un intérprete de Python, elige el de `.venv`.

### 4. Comprobar

```bash
quarto preview
```

Si se abre la web en el navegador, está todo listo.

## Cada vez que vayas a trabajar

```bash
cd proy-redesneuronales
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
git switch test
git pull
```

## Forma de trabajar

- `master` y `test` están protegidas: no se hace push directo, todo entra por pull request.
- Crea una rama corta por tarea desde `test`, con prefijo del bloque (`eda/`, `modelo/`, `experimentos/`, `informe/`, `infra/`) y abre un pull request hacia `test`.
- Los datos originales (`data/raw/`) nunca se suben. Solo el dataset limpio y reducido en `data/processed/`.
- El trabajo de redes neuronales está en la carpeta `redes neuronales/`, un `.qmd` por sección.
