import base64
import datetime
import html
import io
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import sympy as sp
from flask import Flask, render_template_string, request
from sympy.integrals.transforms import LaplaceTransform
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Core symbols/functions
t, s = sp.symbols("t s", real=True)
a, b = sp.symbols("a b", real=True)
x = sp.Function("x")
u = sp.Function("u")

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


TEMPLATE = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Transformadas de Laplace</title>
  <style>
    :root {
      --bg: #eef5ff;
      --panel: #ffffff;
      --ink: #0f2037;
      --muted: #55708f;
      --accent: #2467b3;
      --accent-strong: #1d4f8a;
      --line: #d8e5f4;
      --line-strong: #b7cde7;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at 15% -10%, #ffffff 0%, #edf5ff 42%, var(--bg) 100%),
        linear-gradient(180deg, #f5f9ff, #eaf2fc);
      color: var(--ink);
    }
    .wrap {
      max-width: 1024px;
      margin: 34px auto;
      padding: 0 20px 0;
    }
    .wrap .card:last-child {
      margin-bottom: 0;
    }
    h1 {
      margin: 0;
      font-size: 2rem;
      letter-spacing: -0.02em;
    }
    .lead {
      margin: 10px 0 22px;
      color: var(--muted);
      line-height: 1.5;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px;
      margin: 16px 0;
      box-shadow:
        0 10px 26px rgba(26, 66, 117, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }
    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 4px;
    }
    .card-body.hidden {
      display: none;
    }
    .section-toggle {
      width: 34px !important;
      min-width: 34px;
      height: 34px;
      margin-top: 0 !important;
      padding: 0;
      border-radius: 7px;
      font-size: 1.2rem;
      font-weight: 400;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: #e7f1ff;
      color: var(--accent-strong);
      border: 1px solid #c6dbf4;
    }
    .section-toggle:hover {
      background: #d9e9fb;
    }
    .card h2 {
      margin: 0 0 8px;
      font-size: 1.4rem;
      font-weight: 700;
      color: #1f5ea4;
    }
    .card p {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    label {
      font-weight: 640;
      display: block;
      margin: 10px 0 4px;
    }
    .label-plain {
      font-weight: 400;
    }
    input, textarea, button {
      width: 100%;
      box-sizing: border-box;
      padding: 10px 12px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      font-size: 0.95rem;
      background: #fbfdff;
      color: var(--ink);
    }
    .mono-input {
      font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 0.94rem;
      letter-spacing: 0.01em;
    }
    input:focus, textarea:focus {
      border-color: var(--accent);
      outline: 2px solid rgba(36, 103, 179, 0.17);
      outline-offset: 1px;
    }
    button {
      background: var(--accent);
      color: white;
      font-weight: 650;
      border: none;
      cursor: pointer;
      margin-top: 12px;
      transition: background-color 0.2s ease;
    }
    button:hover {
      background: var(--accent-strong);
    }
    .secondary-btn {
      background: #e7f1ff;
      color: var(--accent-strong);
      border: 1px solid #c6dbf4;
    }
    .secondary-btn:hover {
      background: #d9e9fb;
    }
    .report-btn {
      width: auto;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .result {
      margin-top: 14px;
      padding: 12px 13px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f4f9ff;
      overflow-x: auto;
    }
    .result img {
      display: block;
      width: 100%;
      max-width: 760px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: white;
    }
    .inline-controls {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      align-items: end;
      margin-top: 8px;
      width: 100%;
      max-width: 760px;
    }
    .inline-controls button {
      width: 100%;
      margin-top: 0;
      white-space: normal;
      padding: 10px 16px;
    }
    @media (max-width: 720px) {
      .inline-controls {
        grid-template-columns: 1fr;
      }
      .inline-controls button {
        width: 100%;
      }
    }
    .error {
      color: #9b2226;
      font-weight: 600;
      margin-top: 8px;
    }
    code {
      background: #e7f1ff;
      padding: 2px 5px;
      border-radius: 4px;
      color: #214f86;
    }
    .footer-note {
      text-align: center;
      color: #5f7c9f;
      font-size: 0.9rem;
      margin-top: 0;
      padding: 0 20px 10px;
    }
    .footer-note a {
      color: inherit;
      text-decoration: none;
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
  {% set active_section = form.get('action', 'laplace') %}
  <div class="wrap">
    <h1>Calculadora de transformadas de Laplace</h1>
    <p class="lead">Usa las literales <code>t</code> para el tiempo y <code>s</code> para la frecuencia.</p>

    <div class="card">
      <div class="card-header">
        <h2>Del tiempo a la frecuencia</h2>
        <button type="button" class="section-toggle" data-target="laplace-body">&minus;</button>
      </div>
      <div id="laplace-body" class="card-body{% if active_section not in ['laplace'] %} hidden{% endif %}">
        <p>Calcula la transformada de Laplace de una señal en el tiempo.</p>
        <form method="post">
          <input type="hidden" name="action" value="laplace" />
          <label>Expresión \(x(t)\)</label>
          <input class="mono-input" name="x_expr" value="{{ form.get('x_expr', 'exp(-2*t)*sin(3*t)') }}" placeholder="exp(-2*t)*sin(3*t)" required />
          <button type="submit">Calcular \(\mathscr{L}\)</button>
        </form>
        {% if results.laplace_pair %}
          <div class="result">\[ {{ results.laplace_pair }} \]</div>
        {% endif %}
        {% if errors.laplace %}<div class="error">{{ errors.laplace }}</div>{% endif %}
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2>De la frecuencia al tiempo</h2>
        <button type="button" class="section-toggle" data-target="inverse-body">+</button>
      </div>
      <div id="inverse-body" class="card-body{% if active_section not in ['inverse'] %} hidden{% endif %}">
        <p>Calcula la transformada inversa de Laplace para regresar al dominio temporal.</p>
        <form method="post">
          <input type="hidden" name="action" value="inverse" />
          <label>Expresión \(X(s)\)</label>
          <input class="mono-input" name="X_expr" value="{{ form.get('X_expr', '(2s+1)/(s^2+5s+6)') }}" placeholder="1/(s^2+5*s+6)" required />
          <button type="submit">Calcular \(\mathscr{L}^{-1}\)</button>
        </form>
        {% if results.inverse_pair %}
          <div class="result">
            \[ {{ results.inverse_pair }} \]
            {% for hint in results.inverse_hints %}
              <div>{{ hint|safe }}</div>
            {% endfor %}
          </div>
        {% endif %}
        {% if errors.inverse %}<div class="error">{{ errors.inverse }}</div>{% endif %}
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2>Ecuaciones diferenciales lineales de coeficientes constantes</h2>
        <button type="button" class="section-toggle" data-target="ode-body">+</button>
      </div>
      <div id="ode-body" class="card-body{% if active_section not in ['ode', 'ode_report'] %} hidden{% endif %}">
        <p>Transforma ambos lados de la ecuación diferencial y resuelve para la incógnita usando transformada inversa.</p>
        <form method="post">
          <input type="hidden" name="action" value="ode" />
          <label>Función incógnita</label>
          <input class="mono-input" name="unknown_name" value="{{ form.get('unknown_name', 'x(t)') }}" placeholder="x o x(t)" required />
          <label>Ecuación diferencial</label>
          <textarea class="mono-input" name="ode_expr" rows="3" required>{{ form.get('ode_expr', "x''(t) + 5*x'(t) + 2*x(t) = 3*sin(t)") }}</textarea>
          {% set unk_label = form.get('unknown_name', 'x')|replace('(t)', '')|replace(' ', '') %}
          <label>Condiciones iniciales \([{{ unk_label }}(0), {{ unk_label }}'(0), ...]\)</label>
          <input class="mono-input" name="ic_list" value="{{ form.get('ic_list', '[0, 0]') }}" placeholder="[0, 0]" />
          <button type="submit">Transformar y resolver</button>
        </form>

        {% if results.ode_transform %}
          <div class="result">
            <strong>Ecuación transformada:</strong> \[ {{ results.ode_transform }} \]
            {% if results.required_ics is defined %}
              <div>Modelo de orden {{ results.required_ics }}, se usan {{ results.required_ics }} condiciones iniciales.</div>
            {% endif %}
          </div>
        {% endif %}
        {% if results.ode_with_ics %}
          <div class="result"><strong>Reemplazo de condiciones iniciales:</strong> \[ {{ results.ode_with_ics }} \]</div>
        {% endif %}
        {% if results.X_of_s %}
          <div class="result"><strong>Solución en el dominio de la frecuencia:</strong> \[ {{ results.unknown_transform_name }}(s)={{ results.X_of_s }} \]</div>
        {% endif %}
        {% if results.x_of_t %}
          <div class="result">
            <strong>Solución en el dominio del tiempo:</strong> \[ {{ results.unknown_name }}(t)={{ results.x_of_t }} \]
            {% for hint in results.x_of_t_hints %}
              <div>{{ hint|safe }}</div>
            {% endfor %}
          </div>
        {% endif %}
        {% if results.can_plot %}
          {% if results.plot_requested %}
            <div class="result">
              {% if results.plot_data %}
                <strong style="display:inline-block;margin-bottom:6px;">Gráfica de la solución:</strong><br/>
                <img alt="Grafica de la solucion" src="data:image/png;base64,{{ results.plot_data }}" />
              {% endif %}
              <form method="post">
                <input type="hidden" name="action" value="ode" />
                <input type="hidden" name="unknown_name" value="{{ form.get('unknown_name', 'x') }}" />
                <input type="hidden" name="ode_expr" value="{{ form.get('ode_expr', '') }}" />
                <input type="hidden" name="ic_list" value="{{ form.get('ic_list', '') }}" />
                <input type="hidden" name="plot_now" value="1" />
                <div class="inline-controls">
                  <div>
                    <label class="label-plain">Tiempo inicial</label>
                    <input class="mono-input" name="t_min" value="{{ results.t_min }}" placeholder="0" />
                  </div>
                  <div>
                    <label class="label-plain">Tiempo final</label>
                    <input class="mono-input" name="t_max" value="{{ results.t_max }}" placeholder="10" />
                  </div>
                  <button type="submit">Actualizar gráfica</button>
                </div>
              </form>
            </div>
          {% else %}
            <form method="post">
              <input type="hidden" name="action" value="ode" />
              <input type="hidden" name="unknown_name" value="{{ form.get('unknown_name', 'x') }}" />
              <input type="hidden" name="ode_expr" value="{{ form.get('ode_expr', '') }}" />
              <input type="hidden" name="ic_list" value="{{ form.get('ic_list', '') }}" />
              <input type="hidden" name="plot_now" value="1" />
              <button type="submit">Graficar la solución</button>
            </form>
          {% endif %}
        {% endif %}
        {% if results.plot_note %}
          <div class="result">{{ results.plot_note }}</div>
        {% endif %}
        {% if results.x_of_t %}
          <form method="post" target="_blank">
            <input type="hidden" name="action" value="ode_report" />
            <input type="hidden" name="unknown_name" value="{{ form.get('unknown_name', '') }}" />
            <input type="hidden" name="ode_expr" value="{{ form.get('ode_expr', '') }}" />
            <input type="hidden" name="ic_list" value="{{ form.get('ic_list', '') }}" />
            <input type="hidden" name="t_min" value="{{ results.get('t_min', form.get('t_min', '0')) }}" />
            <input type="hidden" name="t_max" value="{{ results.get('t_max', form.get('t_max', '10')) }}" />
            <button type="submit" class="secondary-btn report-btn">Crear reporte PDF</button>
          </form>
        {% endif %}
        {% if errors.ode %}<div class="error">{{ errors.ode }}</div>{% endif %}
      </div>
    </div>
  </div>
  <footer class="footer-note">
    &copy; 2025, <a href="https://isantosruiz.github.io/home/" target="_blank" rel="noopener noreferrer">Ildeberto de los Santos Ruiz</a>
  </footer>
  <script>
    const MINUS = "\u2212";
    document.querySelectorAll(".section-toggle").forEach((btn) => {
      const body = document.getElementById(btn.dataset.target);
      const syncSymbol = () => {
        if (!body) return;
        btn.textContent = body.classList.contains("hidden") ? "+" : MINUS;
      };
      syncSymbol();
      btn.addEventListener("click", () => {
        if (!body) return;
        body.classList.toggle("hidden");
        syncSymbol();
      });
    });
  </script>
</body>
</html>
"""


def _normalize_primes(expr: str) -> str:
    expr = expr.replace("^", "**")

    def replace_primes(match: re.Match[str]) -> str:
        fname = match.group(1)
        primes = match.group(2)
        n = len(primes)
        if n == 1:
            return f"Derivative({fname}(t), t)"
        return f"Derivative({fname}(t), (t,{n}))"

    return re.sub(r"\b([A-Za-z_]\w*)('+?)\(t\)", lambda m: replace_primes(m) if m.group(2) else m.group(0), expr)


def _base_local_dict() -> dict:
    return {
        "t": t,
        "s": s,
        "a": a,
        "b": b,
        "x": x,
        "u": u,
        "Derivative": sp.Derivative,
        "Heaviside": sp.Heaviside,
        "DiracDelta": sp.DiracDelta,
        "exp": sp.exp,
        "sin": sp.sin,
        "sen": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "log10": lambda x: sp.log(x, 10),
        "pi": sp.pi,
        "E": sp.E,
        "e": sp.E,
    }


def _parse_math(expr: str, extra_locals: dict | None = None):
    prepared = _normalize_primes(expr)
    local_dict = _base_local_dict()
    if extra_locals:
        local_dict.update(extra_locals)
    return parse_expr(prepared, local_dict=local_dict, transformations=TRANSFORMATIONS)


def _parse_math_keep_order(expr: str, extra_locals: dict | None = None):
    prepared = _normalize_primes(expr)
    local_dict = _base_local_dict()
    if extra_locals:
        local_dict.update(extra_locals)
    return parse_expr(
        prepared,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=False,
    )


def _parse_equation(expr: str, extra_locals: dict | None = None) -> sp.Eq:
    if "=" not in expr:
        lhs, rhs = expr, "0"
    else:
        lhs, rhs = expr.split("=", 1)
    return sp.Eq(_parse_math(lhs, extra_locals), _parse_math(rhs, extra_locals))


def _clean_latex(tex: str) -> str:
    tex = re.sub(r"\\theta\s+\\left", r"\\theta\\left", tex)
    tex = re.sub(r"\\theta\s+\(", r"\\theta(", tex)
    tex = tex.replace(r"\theta\left(t\right)", r"\theta(t)")
    tex = re.sub(r"([A-Za-z_]\w*)\^\{\((\d+)\)\(0\)\}", r"\1^{(\2)}(0)", tex)
    tex = re.sub(
        r"([A-Za-z_]\w*)\^\{\\left\((\d+)\\right\)\\left\(0\\right\)\}",
        r"\1^{(\2)}(0)",
        tex,
    )
    tex = re.sub(r"([A-Z][A-Za-z0-9_]*\(s\))\s+s\^\{([^}]+)\}", r"s^{\2} \1", tex)
    tex = re.sub(r"([A-Z][A-Za-z0-9_]*\(s\))\s+s(?![A-Za-z])", r"s \1", tex)
    return tex


def _to_latex(expr) -> str:
    return _clean_latex(sp.latex(sp.simplify(expr)))


def _laplace_pair_latex(time_expr, freq_expr) -> str:
    return r"\mathscr{L}\left\{" + _to_latex(time_expr) + r"\right\} = " + _to_latex(freq_expr)


def _inverse_pair_latex(freq_expr, time_expr) -> str:
    return r"\mathscr{L}^{-1}\left\{" + _to_latex(freq_expr) + r"\right\} = " + _to_latex(time_expr)


def _inverse_hints(expr) -> list[str]:
    has_dirac = expr.has(sp.DiracDelta)
    has_heaviside = expr.has(sp.Heaviside)
    if has_dirac and has_heaviside:
        return [
            r"\(\delta(t)\) es la delta de Dirac: un impulso ideal en \(t=0\) con área unitaria. "
            r"\(\theta(t)\) es la función escalón de Heaviside: vale 0 para \(t<0\) y 1 para \(t\ge 0\)."
        ]
    if has_dirac:
        return [r"\(\delta(t)\) es la delta de Dirac: un impulso ideal en \(t=0\) con área unitaria."]
    if has_heaviside:
        return [r"\(\theta(t)\) es la función escalón de Heaviside: vale 0 para \(t<0\) y 1 para \(t\ge 0\)."]
    return []


def _validate_unknown_name(name: str) -> str:
    clean = name.strip() or "x"
    m = re.fullmatch(r"([A-Za-z_]\w*)\s*\(\s*t\s*\)", clean)
    if m:
        clean = m.group(1)
    if not re.fullmatch(r"[A-Za-z_]\w*", clean):
        raise ValueError("El nombre de la función desconocida debe ser un identificador válido, por ejemplo x o y.")
    return clean


def _split_top_level_commas(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        current.append(ch)
    last = "".join(current).strip()
    if last:
        items.append(last)
    return items


def _split_top_level_add_terms(text: str) -> list[str]:
    terms: list[str] = []
    current: list[str] = []
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in "+-" and i > 0:
            prev = text[i - 1]
            if prev not in "eE":
                terms.append("".join(current).strip())
                current = [ch]
                continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        terms.append(tail)
    return terms


def _ordered_latex_from_input(text: str, extra_locals: dict | None = None) -> str:
    parts = _split_top_level_add_terms(text)
    if not parts:
        return _clean_latex(sp.latex(_parse_math_keep_order(text, extra_locals=extra_locals)))

    out: list[str] = []
    for i, part in enumerate(parts):
        token = part.strip()
        sign = "+"
        if token.startswith("+"):
            token = token[1:].strip()
        elif token.startswith("-"):
            sign = "-"
            token = token[1:].strip()

        term_latex = _latex_derivatives_as_primes(
            _clean_latex(sp.latex(_parse_math_keep_order(token, extra_locals=extra_locals)))
        )
        if i == 0:
            out.append(("-" if sign == "-" else "") + term_latex)
        else:
            out.append((" - " if sign == "-" else " + ") + term_latex)
    return "".join(out)


def _latex_derivatives_as_primes(tex: str) -> str:
    def repl_n(match: re.Match[str]) -> str:
        order = int(match.group(1))
        fname = match.group(2)
        primes = r"\prime" * order
        return rf"{fname}^{{{primes}}}\left(t\right)"

    tex = re.sub(
        r"\\frac\{d\^\{(\d+)\}\}\{d t\^\{\1\}\}\s*([A-Za-z_]\w*)\{\\left\(t \\right\)\}",
        repl_n,
        tex,
    )
    tex = re.sub(
        r"\\frac\{d\}\{d t\}\s*([A-Za-z_]\w*)\{\\left\(t \\right\)\}",
        lambda m: rf"{m.group(1)}^{{\prime}}\left(t\right)",
        tex,
    )
    return tex


def _ordered_transformed_side_latex(
    raw_side: str,
    fname: str,
    unknown_fn,
    unknown_s,
    ic_term_subs: dict,
) -> str:
    parts = _split_top_level_add_terms(raw_side)
    if not parts:
        expr = _parse_math(raw_side, extra_locals={fname: unknown_fn})
        side = sp.laplace_transform(expr, t, s, noconds=True).xreplace({
            LaplaceTransform(unknown_fn(t), t, s): unknown_s,
        })
        side = side.xreplace(ic_term_subs)
        return _to_latex(side)

    out: list[str] = []
    for i, part in enumerate(parts):
        token = part.strip()
        sign = "+"
        if token.startswith("+"):
            token = token[1:].strip()
        elif token.startswith("-"):
            sign = "-"
            token = token[1:].strip()

        term_expr = _parse_math(token, extra_locals={fname: unknown_fn})
        term_s = sp.laplace_transform(term_expr, t, s, noconds=True).xreplace({
            LaplaceTransform(unknown_fn(t), t, s): unknown_s,
        })
        term_s = term_s.xreplace(ic_term_subs)
        term_tex = _to_latex(term_s)
        if i == 0:
            out.append(("-" if sign == "-" else "") + term_tex)
        else:
            out.append((" - " if sign == "-" else " + ") + term_tex)
    return "".join(out)


def _derivative_order(expr, fn) -> int:
    order = 0
    for der in expr.atoms(sp.Derivative):
        if der.expr == fn(t):
            n = sum(1 for var in der.variables if var == t)
            order = max(order, n)
    return order


def _ic_placeholder(k: int, fname: str):
    if k == 0:
        return sp.Symbol(f"{fname}(0)")
    if k == 1:
        return sp.Symbol(f"{fname}'(0)")
    return sp.Symbol(f"{fname}^({k})(0)")


def _parse_ic_list(raw: str, expected: int, extra_locals: dict) -> list[sp.Expr]:
    text = (raw or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("Las condiciones iniciales deben ingresarse como lista, por ejemplo [0,1].")
    inner = text[1:-1].strip()
    if not inner:
        values: list[sp.Expr] = []
    else:
        parts = _split_top_level_commas(inner)
        values = [_parse_math(part, extra_locals=extra_locals) for part in parts]
    if len(values) != expected:
        raise ValueError(
            f"La EDO es de orden {expected}. Debes proporcionar exactamente {expected} condiciones iniciales en la lista."
        )
    return values


def _plot_solution(expr, unknown_name: str, t_min: float = 0.0, t_max: float = 10.0):
    if expr.has(sp.DiracDelta):
        return None
    if expr.free_symbols - {t}:
        return None

    try:
        fn = sp.lambdify(t, sp.simplify(expr), modules=["numpy"])
        xs = np.linspace(t_min, t_max, 500)
        ys = np.asarray(fn(xs), dtype=np.complex128)
        if ys.shape == ():
            ys = np.full_like(xs, ys, dtype=np.complex128)
        if np.max(np.abs(ys.imag)) > 1e-8:
            return None
        ys = ys.real
        mask = np.isfinite(ys)
        if not np.any(mask):
            return None

        fig, ax = plt.subplots(figsize=(8.2, 3.4), dpi=140)
        ax.plot(xs[mask], ys[mask], color="#1f3a5f", linewidth=2.0)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_xlabel("t")
        ax.set_ylabel(f"{unknown_name}(t)")
        ax.grid(True, alpha=0.24)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception:
        return None


def _solve_ode_action(form) -> tuple[dict, dict]:
    results: dict = {}

    fname = _validate_unknown_name(form.get("unknown_name", "x"))
    unknown_fn = sp.Function(fname)
    transform_name = fname[0].upper() + fname[1:]
    unknown_s = sp.Symbol(f"{transform_name}(s)")
    ode_raw = form.get("ode_expr", "")

    ode = _parse_equation(ode_raw, extra_locals={fname: unknown_fn})
    ode_order = max(_derivative_order(ode.lhs, unknown_fn), _derivative_order(ode.rhs, unknown_fn))
    results["required_ics"] = ode_order

    ic_raw = form.get("ic_list", "")
    ic_values = _parse_ic_list(ic_raw, ode_order, extra_locals={fname: unknown_fn})
    ic_term_subs = {}
    ic_value_subs = {}
    for k in range(ode_order):
        if k == 0:
            original = unknown_fn(0)
        else:
            original = sp.Subs(sp.Derivative(unknown_fn(t), (t, k)), t, 0)
        placeholder = _ic_placeholder(k, fname)
        ic_term_subs[original] = placeholder
        ic_value_subs[placeholder] = ic_values[k]

    lhs_s = sp.laplace_transform(ode.lhs, t, s, noconds=True)
    rhs_s = sp.laplace_transform(ode.rhs, t, s, noconds=True)
    ode_s = sp.Eq(lhs_s, rhs_s).xreplace({
        LaplaceTransform(unknown_fn(t), t, s): unknown_s,
    })
    ode_s = ode_s.xreplace(ic_term_subs)

    if "=" in ode_raw:
        lhs_raw, rhs_raw = ode_raw.split("=", 1)
    else:
        lhs_raw, rhs_raw = ode_raw, "0"
    lhs_display = _ordered_latex_from_input(lhs_raw, extra_locals={fname: unknown_fn})
    rhs_display = _ordered_latex_from_input(rhs_raw, extra_locals={fname: unknown_fn})
    lhs_transformed = _ordered_transformed_side_latex(lhs_raw, fname, unknown_fn, unknown_s, ic_term_subs)
    rhs_transformed = _ordered_transformed_side_latex(rhs_raw, fname, unknown_fn, unknown_s, ic_term_subs)
    laplace_step = (
        r"\mathscr{L}\left\{" + lhs_display + r"\right\} = "
        r"\mathscr{L}\left\{" + rhs_display + r"\right\}"
    )
    results["ode_transform"] = (
        r"\begin{aligned}"
        + laplace_step
        + r"\\ \Rightarrow\quad "
        + lhs_transformed
        + r" = "
        + rhs_transformed
        + r"\end{aligned}"
    )

    ode_s_ic = ode_s.subs(ic_value_subs)
    results["ode_with_ics"] = _clean_latex(sp.latex(ode_s_ic))

    x_sol = sp.solve(ode_s_ic, unknown_s)
    if not x_sol:
        raise ValueError("No se pudo despejar la transformada de la función desconocida.")

    X_expr = sp.simplify(x_sol[0])
    x_time = sp.inverse_laplace_transform(X_expr, s, t)

    results["unknown_name"] = fname
    results["unknown_transform_name"] = transform_name
    results["X_of_s"] = _to_latex(X_expr)
    results["x_of_t"] = _to_latex(x_time)
    results["x_of_t_hints"] = _inverse_hints(x_time)

    results["can_plot"] = not x_time.has(sp.DiracDelta) and not (x_time.free_symbols - {t})
    results["plot_requested"] = form.get("plot_now", "") == "1"
    t_min_raw = (form.get("t_min", "") or "").strip()
    t_max_raw = (form.get("t_max", "") or "").strip()
    t_min = 0.0 if t_min_raw == "" else float(t_min_raw)
    t_max = 10.0 if t_max_raw == "" else float(t_max_raw)
    results["t_min"] = sp.nsimplify(t_min) if t_min.is_integer() else t_min
    results["t_max"] = sp.nsimplify(t_max) if t_max.is_integer() else t_max
    if not t_max > t_min:
        raise ValueError("El límite superior de tiempo debe ser mayor que el límite inferior.")

    if results["plot_requested"]:
        plot_data = _plot_solution(x_time, fname, t_min=t_min, t_max=t_max)
        if plot_data:
            results["plot_data"] = plot_data

    if results["plot_requested"] and (x_time.free_symbols - {t}):
        results["plot_note"] = (
            r"La solución depende de parámetros simbólicos además de \(t\); "
            r"se requieren valores numéricos para poder graficar."
        )

    report_data = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "unknown_name": fname,
        "unknown_transform_name": transform_name,
        "ode_expr": ode_raw,
        "ode_input_latex": f"{lhs_display} = {rhs_display}",
        "ic_list": ic_raw,
        "ic_list_latex": (
            r"\left[" + ", ".join(_to_latex(value) for value in ic_values) + r"\right]"
            if ic_values
            else r"\left[\right]"
        ),
        "required_ics": ode_order,
        "ode_transform": results["ode_transform"],
        "ode_with_ics": results["ode_with_ics"],
        "X_of_s": results["X_of_s"],
        "x_of_t": results["x_of_t"],
        "t_min": t_min,
        "t_max": t_max,
        "plot_data": results.get("plot_data"),
    }

    if report_data["plot_data"] is None and results["can_plot"]:
        report_data["plot_data"] = _plot_solution(x_time, fname, t_min=t_min, t_max=t_max)

    return results, report_data


def _escape_html(value) -> str:
    return html.escape(str(value), quote=True)


def _report_math_line(latex_text: str) -> str:
    return f'<div class="report-math">\\[{latex_text}\\]</div>'


def _report_section(title: str, content_html: str) -> str:
    return f'<section class="report-section"><h2>{title}</h2>{content_html}</section>'


def _build_ode_report_html(report_data: dict) -> str:
    unknown_name = _escape_html(report_data["unknown_name"])
    unknown_transform_name = _escape_html(report_data["unknown_transform_name"])
    generated_at = _escape_html(report_data["generated_at"])
    required_ics = _escape_html(report_data["required_ics"])
    t_min = _escape_html(report_data["t_min"])
    t_max = _escape_html(report_data["t_max"])

    input_section = _report_section(
        "Datos de entrada",
        "".join(
            [
                '<div class="report-line"><strong>Función incógnita:</strong></div>',
                _report_math_line(f"{unknown_name}(t)"),
                '<div class="report-line"><strong>Ecuación diferencial:</strong></div>',
                _report_math_line(report_data["ode_input_latex"]),
                '<div class="report-line"><strong>Condiciones iniciales:</strong></div>',
                _report_math_line(report_data["ic_list_latex"]),
                '<div class="report-line"><strong>Condiciones requeridas:</strong></div>',
                _report_math_line(str(required_ics)),
            ]
        ),
    )

    process_steps = "".join(
        [
            _report_section("1. Ecuación transformada", _report_math_line(report_data["ode_transform"])),
            _report_section("2. Reemplazo de condiciones iniciales", _report_math_line(report_data["ode_with_ics"])),
            _report_section(
                "3. Solución en el dominio de la frecuencia",
                _report_math_line(f"{unknown_transform_name}(s) = {report_data['X_of_s']}"),
            ),
            _report_section(
                "4. Solución en el dominio del tiempo",
                _report_math_line(f"{unknown_name}(t) = {report_data['x_of_t']}"),
            ),
        ]
    )

    plot_html = ""
    if report_data.get("plot_data"):
        plot_html = _report_section(
            "5. Gráfica de la solución",
            (
                f'<div class="report-figure"><img alt="Grafica de la solucion" src="data:image/png;base64,{report_data["plot_data"]}" /></div>'
                f'<div class="report-line">Rango graficado: t = {t_min} a t = {t_max}</div>'
            ),
        )

    return f"""<!doctype html>
<html lang="es">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Reporte de transformada de Laplace</title>
    <style>
      :root {{
        --bg: #edf5ff;
        --panel: #ffffff;
        --ink: #10243c;
        --muted: #56708f;
        --line: #d7e5f3;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
        color: var(--ink);
      }}
      .toolbar {{
        position: sticky;
        top: 0;
        z-index: 5;
        padding: 12px 16px;
        background: rgba(248, 252, 255, 0.92);
        border-bottom: 1px solid var(--line);
      }}
      .toolbar button {{
        border: 1px solid #2b6cb0;
        border-radius: 8px;
        padding: 8px 14px;
        background: #2b6cb0;
        color: #fff;
        font-weight: 600;
        cursor: pointer;
      }}
      .report {{
        max-width: 960px;
        margin: 0 auto;
        padding: 16px 18px 24px;
      }}
      h1 {{
        margin: 0;
        color: #1f5ea4;
      }}
      .meta {{
        margin-top: 6px;
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .report-section {{
        margin-top: 16px;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 12px 14px;
        background: #f6faff;
      }}
      .report-section h2 {{
        margin: 0 0 8px;
        font-size: 1.05rem;
        color: #1f5ea4;
      }}
      .report-line {{
        margin: 8px 0 4px;
      }}
      .report-math {{
        margin: 4px 0 8px;
        overflow-x: auto;
      }}
      .report-figure {{
        margin-top: 8px;
      }}
      .report-figure img {{
        width: 100%;
        border-radius: 8px;
        border: 1px solid var(--line);
        background: white;
      }}
      .report-footer {{
        margin-top: 16px;
        text-align: center;
        color: var(--muted);
        font-size: 0.85rem;
        padding-top: 6px;
      }}
      .report-footer a {{
        color: #1f5ea8;
        text-decoration: none;
      }}
      @media print {{
        .toolbar {{ display: none; }}
        .report {{
          max-width: none;
          margin: 0;
          padding: 8mm 8mm 10mm;
        }}
        .report-section {{
          break-inside: avoid;
        }}
      }}
    </style>
    <script>
      window.MathJax = {{
        tex: {{ inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]] }},
        svg: {{ fontCache: "global" }},
      }};
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <script>
      window.addEventListener("load", () => {{
        const runPrint = () => setTimeout(() => window.print(), 250);
        if (window.MathJax?.typesetPromise) {{
          window.MathJax.typesetPromise().then(runPrint).catch(runPrint);
          return;
        }}
        runPrint();
      }});
    </script>
  </head>
  <body>
    <div class="toolbar">
      <button type="button" onclick="window.print()">Imprimir / Guardar PDF</button>
    </div>
    <main class="report">
      <h1>Reporte de transformada de Laplace</h1>
      <div class="meta">Generado: {generated_at}</div>
      {input_section}
      {_report_section("Proceso paso a paso", process_steps)}
      {plot_html}
      <footer class="report-footer">
        &copy; 2026, <a href="https://isantosruiz.github.io/home/" target="_blank" rel="noopener noreferrer">Ildeberto de los Santos Ruiz</a>
      </footer>
    </main>
  </body>
</html>"""


@app.route("/", methods=["GET", "POST"])
def home():
    results = {}
    errors = {}
    form = request.form

    if request.method == "POST":
        action = form.get("action", "")

        if action == "laplace":
            try:
                x_expr = _parse_math(form.get("x_expr", ""))
                X_expr = sp.laplace_transform(x_expr, t, s, noconds=True)
                results["laplace_pair"] = _laplace_pair_latex(x_expr, X_expr)
            except Exception as exc:
                errors["laplace"] = f"Error al transformar x(t): {exc}"

        elif action == "inverse":
            try:
                X_expr = _parse_math(form.get("X_expr", ""))
                x_expr = sp.inverse_laplace_transform(X_expr, s, t)
                results["inverse_pair"] = _inverse_pair_latex(X_expr, x_expr)
                results["inverse_hints"] = _inverse_hints(x_expr)
            except Exception as exc:
                errors["inverse"] = f"Error al calcular la inversa: {exc}"

        elif action in {"ode", "ode_report"}:
            try:
                ode_results, report_data = _solve_ode_action(form)
                results.update(ode_results)

                if action == "ode_report":
                    report_html = _build_ode_report_html(report_data)
                    return app.response_class(
                        response=report_html,
                        status=200,
                        mimetype="text/html",
                    )

            except Exception as exc:
                errors["ode"] = f"Error en EDO: {exc}"

    return render_template_string(TEMPLATE, results=results, errors=errors, form=form)


# Vercel looks for `app` in Python serverless files.
if __name__ == "__main__":
    app.run(debug=True)
