# Laplace App (Python + Vercel)

Aplicación web en Python (Flask + SymPy) para:

- Calcular transformada de Laplace: `x(t) -> X(s)`
- Calcular transformada inversa: `X(s) -> x(t)`
- Transformar ecuaciones diferenciales en ambos lados
- Elegir el nombre de la función desconocida (`x`, `y`, etc.)
- Ingresar condiciones iniciales como lista: `[x(0), x'(0), ...]`
- Resolver la EDO por Laplace inversa y graficar cuando la solución depende solo de `t`

## Ejecutar localmente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api/index.py
```

Abrir `http://127.0.0.1:5000`.

## Despliegue en Vercel

Este repositorio ya incluye `vercel.json` para desplegar `api/index.py` con `@vercel/python`.
