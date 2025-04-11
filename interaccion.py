import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

visibilidad = ["Visible"] * 20 + ["No visible"] * 20
exito_confederado = ["Éxito"] * 10 + ["Fracaso"] * 10 + ["Éxito"] * 10 + ["Fracaso"] * 10
puntaje = [
    70.0, 74.7, 77.1, 70.0, 70.1, 74.4, 70.0, 76.3, 75.3, 73.1,  # Visible - Éxito
    149.4, 148.3, 147.5, 148.6, 147.2, 144.5, 149.4, 146.3, 146.9, 143.9,  # Visible - Fracaso
    90.3, 87.0, 84.5, 90.5, 88.6, 90.7, 92.8, 92.0, 86.7, 92.9,  # No Visible - Éxito
    75.4, 71.9, 75.9, 67.9, 73.6, 76.3, 74.6, 73.0, 69.4, 67.0   # No Visible - Fracaso
]

df = pd.DataFrame({
    "Visibilidad": visibilidad,
    "ExitoConfederado": exito_confederado,
    "Puntaje": puntaje
})

df["Grupo"] = df["Visibilidad"] + " - " + df["ExitoConfederado"]

modelo = ols('Puntaje ~ C(Visibilidad) * C(ExitoConfederado)', data=df).fit()
anova_table = sm.stats.anova_lm(modelo, typ=2)

print("\nResultados del ANOVA:")
print(anova_table)

tukey = pairwise_tukeyhsd(df["Puntaje"],    ["Grupo"], alpha=0.05)

print("\nResultados de la Prueba de Tukey:")
print(tukey)
