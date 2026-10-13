"""Spanish text of the feature selection report.

Used by src/06b_report.py when it is run with ``--lang es``. Only the
explanatory text is translated. Feature names, labels, configuration names,
file paths, metric names and library names stay exactly as they appear in the
code and the CSV files. The placeholders in braces are the same ones the
English template uses, so both versions are filled with the same numbers.
"""

TEMPLATE_ES = """# Selección de características con métodos de filtro para un conjunto de datos de flujos de un IDS

Proyecto "Arquitectura IDS impulsada por IA". Autora: Laura Camila Herrera
Murillo. Fecha: {date}.

Todos los números de este informe se leen de los archivos de
resultados en `results/`; los scripts que los producen se listan en la
sección 9. Al final hay un glosario de los términos técnicos.

## 1. Resumen

- Se aplicaron tres métodos de filtro a dos conjuntos de datos de flujos de
  red (A, capturado con firewall; B, sin firewall): Low Variance (baja
  varianza), correlación de Pearson y Chi-cuadrado. Cada filtro se ajustó
  solo con las filas de entrenamiento (train) y cada subconjunto de
  características (features, las columnas de entrada del modelo) resultante
  se evaluó con el mismo RandomForest (modelo de clasificación formado por
  muchos árboles de decisión).
- Ningún filtro mejora el modelo. Lo que logran los filtros es una entrada
  más pequeña sin costo medible: las listas finales tienen
  **{k_A} características para A** y **{k_B} para B** (de 77) y obtienen un
  F1-macro (promedio del F1 de todas las clases, con el mismo peso para cada
  una) de {final_test_A} y {final_test_B} en las filas de prueba (test),
  frente a {base_test_A} y {base_test_B} con todas las características. Las
  diferencias son menores que el ruido entre folds (particiones de la
  validación cruzada) medido en la validación cruzada (error estándar de
  aproximadamente {se_A}).
- El pipeline combinado (secuencia de pasos: Low Variance, luego Pearson,
  luego Chi-cuadrado) alcanza al baseline (modelo de referencia con todas
  las características) con menos características que Chi-cuadrado solo
  ({k_A} frente a 40 en A, {k_B} frente a 40 en B), porque Chi-cuadrado solo
  llena los primeros puestos con características redundantes.
- **Los puntajes no deben leerse como capacidad de detectar ataques.** Las
  características identifican casi a la perfección de cuál de las dos fuentes
  de datos proviene una fila, y las etiquetas que nombran el mismo ataque en
  las dos fuentes nunca se confunden entre sí. La sección 8 explica esta y
  las demás limitaciones.

## 2. Datos y preparación

Cada conjunto de datos une filas de un conjunto externo (IDS2025) con flujos
capturados en un laboratorio virtual, extraídos con `cicflowmeter`: 79
características y la variable objetivo `{target}` con 12 etiquetas. A y B se
procesan por separado con el mismo código.

| Paso | A | B |
|---|---|---|
| Filas originales | {raw_A} | {raw_B} |
| Duplicados exactos eliminados (después de quitar las columnas de puertos) | {dups_A} | {dups_B} |
| Filas limpias | {clean_A} | {clean_B} |
| Filas de entrenamiento (estratificado, {test_pct}% para prueba, semilla {seed}) | {train_A} | {train_B} |
| Filas de prueba | {test_A} | {test_B} |

La limpieza eliminó las dos columnas de puertos (que funcionan como atajos),
las filas con NaN (valor faltante) o Inf (valor infinito), los duplicados
exactos y las filas con etiquetas contradictorias, y dejó 77 características.
Los detalles y las justificaciones están en `cleaning_notes.md` y
`decisions.md`.

## 3. Qué hace cada filtro

Los filtros puntúan las características con estadísticas calculadas a partir
de los datos. No son modelos y no se entrena nada. Los tres se ajustaron solo
con las filas de entrenamiento.

**Low Variance.** Una característica que toma casi el mismo valor en todas
las filas no puede separar clases. La varianza sin escalar depende de la
unidad de cada característica, así que la regla usada es la proporción de
filas que ocupa el valor más frecuente: una característica se elimina cuando
un solo valor cubre al menos el 99.9% de las filas de entrenamiento. Esto
elimina {var_removed_A} características en ambos conjuntos de datos: las 5
columnas constantes y 4 contadores de banderas (flags) con entre 1 y 4 filas
distintas de cero. Un umbral más laxo (99%) se rechazó porque elimina
`Fwd PSH Flags` y `SYN Flag Count`, que son distintas de cero en el 22.5% de
las filas de `Brute Force`: con este desbalance de clases, una característica
casi constante puede ser la señal de una clase pequeña. La varianza después
del escalado MinMax (llevar cada columna al rango de 0 a 1) se calculó y se
descartó como regla, porque penaliza las características con valores extremos
y no las que carecen de información (`decisions.md`, decisión 6).

**Correlación de Pearson.** Dos características con una correlación absoluta
cercana a 1 contienen la misma información. De cada grupo de características
correlacionadas por encima del umbral (se probaron 0.90, 0.95 y 0.99) se
conserva solo una: la que tiene el mayor puntaje ANOVA F (prueba que mide
cuánto separa una característica a las clases) frente a la etiqueta. Las
características constantes no tienen una correlación definida, por lo que
este filtro no puede eliminarlas.

**Chi-cuadrado.** Mide la asociación entre una característica y la etiqueta
en una tabla de conteos. Las características son continuas, así que primero
cada una se corta en {bins} bins (tramos) por cuantiles, con límites
aprendidos en el entrenamiento; sin este paso el puntaje dependería de la
unidad de medida. Las características se ordenan por el estadístico y se
conservan las k mejores (top k, con k = 10, 20, 30, 40). Los puntajes se
contrastaron con la prueba clásica de tabla de contingencia de SciPy para
cada característica. Como comparación se evaluó un segundo ordenamiento por
la V de Cramér (el mismo estadístico corregido por el tamaño de la muestra y
la cantidad de bins).

**Pipeline combinado.** Low Variance al 99.9%, luego Pearson sobre las
características que sobreviven, luego las k mejores de las que sobreviven
según Chi-cuadrado.

## 4. Protocolo de evaluación

- Modelo: `RandomForestClassifier` con {trees} árboles y
  `class_weight="balanced"` (pesos de clase inversos a su frecuencia),
  idéntico para todos los subconjuntos y sin ajuste de hiperparámetros.
- Métricas: F1-macro, PR-AUC macro (área bajo la curva de precisión y
  recall, promediada entre clases), MCC (coeficiente de correlación de
  Matthews, que usa toda la matriz de confusión) y recall (proporción de
  filas de una clase que el modelo detecta) por clase. La exactitud
  (accuracy) no se reporta como métrica principal porque las dos clases más
  grandes reúnen cerca del 70% de las filas.
- Las configuraciones se comparan con validación cruzada estratificada de
  {folds} folds dentro del entrenamiento; los filtros se vuelven a ajustar en
  cada fold.
- Regla de selección, fijada antes de ver los resultados: dentro de cada
  familia, la configuración con menos características cuyo F1-macro promedio
  esté a un error estándar o menos de la mejor.
- Las filas de prueba se usan una sola vez, para el baseline y para la
  configuración seleccionada de cada familia.

## 5. Comparación

### 5.1 Validación cruzada dentro del entrenamiento (todas las configuraciones)

Promedio de {folds} folds; el F1-macro se muestra con su error estándar.
`combined_pX_kY` significa umbral de Pearson X y las Y mejores por
Chi-cuadrado.

{cv_table}

![Comparación en validación cruzada, conjunto de datos A](figures/cv_comparison_A.png)
![Comparación en validación cruzada, conjunto de datos B](figures/cv_comparison_B.png)

### 5.2 Prueba (baseline y la configuración seleccionada de cada familia)

{test_table}

### 5.3 Recall por clase en prueba, baseline frente a la lista final

{recall_table}

Lectura de los resultados:

- Low Variance y Pearson eliminan entre 9 y 32 características sin un cambio
  consistente: la mayor diferencia con el baseline en validación cruzada es
  de 0.010 en F1-macro (Pearson 0.90 en B, error estándar 0.008) y no aparece
  en prueba. Las características redundantes no perjudicaban al baseline: un
  conjunto de árboles las tolera.
- Chi-cuadrado solo necesita 40 características para alcanzar al baseline;
  con 10 o 20 es claramente peor.
- El ordenamiento por la V de Cramér no es una mejor alternativa: con 10
  características su F1-macro cae a {cv10_cramers} y no detecta ninguna fila
  de `sqli`.
- El mayor error individual del baseline es {worst_A} en A y {worst_B}
  en B.

## 6. Por qué el pipeline combinado supera a Chi-cuadrado solo

Chi-cuadrado puntúa cada característica por separado. Dos características que
son copia una de la otra reciben el mismo puntaje alto y ambas entran en las
k mejores, aunque la segunda no aporta nada. Evidencia, para la misma
cantidad de características:

| | A (k = {k_A}) | B (k = {k_B}) |
|---|---|---|
| F1-macro en validación cruzada, Chi-cuadrado top k | {chi_k_A} | {chi_k_B} |
| F1-macro en validación cruzada, combinado top k | {comb_k_A} | {comb_k_B} |
| Pares de características con correlación absoluta mayor que 0.90 dentro de Chi-cuadrado top k | {pairs_top_A} | {pairs_top_B} |
| Lo mismo, dentro de la lista combinada | {pairs_final_A} | {pairs_final_B} |
| Características presentes tanto en Chi-cuadrado top k como en la lista combinada | {overlap_A} | {overlap_B} |

Los primeros puestos del ordenamiento por Chi-cuadrado están ocupados por
grupos como `Total Length of Fwd Packets` y `Subflow Fwd Bytes`, o
`Fwd Packet Length Mean` y `Avg Fwd Segment Size`, que son la misma magnitud
con dos nombres. El pipeline combinado deja que Pearson conserve primero una
característica por grupo, de modo que cada uno de los k puestos corresponde a
información distinta.

## 7. Listas finales de características

Elegidas por la regla de selección: `{final_A}` para A y `{final_B}` para B.
La regla no se ajustó después de ver los resultados (`decisions.md`,
decisión 8), y por eso las dos listas tienen longitudes distintas. Las listas
se ajustaron con el conjunto de entrenamiento completo de cada conjunto de
datos y están guardadas en `results/final_features_A.csv` y
`results/final_features_B.csv`.

### Conjunto de datos A ({k_A} características)

{features_A}

### Conjunto de datos B ({k_B} características)

{features_B}

### A frente a B

- Compartidas ({n_shared}): {shared}
- Solo en A ({n_only_A}): {only_A}
- Solo en B ({n_only_B}): {only_B}

![Matriz de confusión en prueba, conjunto de datos A, lista final](figures/confusion_A_{final_A}.png)

![Matriz de confusión en prueba, conjunto de datos B, lista final](figures/confusion_B_{final_B}.png)

En las matrices de confusión (tablas que cruzan la clase real con la clase
predicha) las etiquetas están ordenadas por fuente de datos. Todos los
errores de clasificación caen dentro del bloque de una sola fuente o
involucran a `benign`, la única etiqueta presente en las dos fuentes.

## 8. Limitaciones

### 8.1 Las características identifican la fuente de los datos

Las filas externas y las del laboratorio provienen de entornos distintos y de
extractores de flujos distintos, y los vocabularios de etiquetas difieren
(`Brute Force` y `bruteforce`, `PortScan` y `recon`, `Web Attack` y `sqli`).
Se hicieron dos comprobaciones:

- **Confusiones entre etiquetas equivalentes:** {pair_errors} en total, en
  las {n_test_configs} configuraciones evaluadas en prueba, en cualquiera de
  los dos sentidos. Si las características describieran el ataque en sí, los
  flujos del mismo tipo de ataque de las dos fuentes se confundirían al menos
  ocasionalmente.
- **Prueba de la fuente:** se entrenó un RandomForest con la misma
  configuración para responder únicamente "¿externo o laboratorio?", usando
  solo filas `benign`, la única etiqueta presente en las dos fuentes. Con las
  listas finales clasifica mal {probe_err_A} de {probe_rows_A} filas de
  prueba en A y {probe_err_B} de {probe_rows_B} en B.

Parte de la razón se ve en las características: {zero_all_A} de las 77
características valen cero en todas las filas del laboratorio (ver 8.3), así
que cualquier valor distinto de cero marca una fila externa. La lista final
de A contiene {zero_final_n_A} de ellas ({zero_final_A}) y la lista final de
B contiene {zero_final_n_B} ({zero_final_B}).

Consecuencia: el modelo separa las fuentes y después las clases dentro de
cada fuente. Los puntajes altos no demuestran que las características
seleccionadas caractericen un tipo de ataque, y no dicen nada sobre el
tráfico de otra red. El experimento que lo pondría a prueba, entrenar con una
fuente y evaluar con las etiquetas equivalentes de la otra, no se realizó.

### 8.2 Infiltration

`Infiltration` tiene 29 filas en total: {inf_train_A} en entrenamiento y
{inf_test_A} en prueba. En prueba el modelo detecta {inf_hits_A} de
{inf_test_A} en A y {inf_hits_B} de {inf_test_B} en B, en todas las
configuraciones; una fila equivale a 17 puntos de recall. En validación
cruzada, sobre las {inf_train_A} filas de entrenamiento, el recall va de
{inf_cv_A} en A y de {inf_cv_B} en B según la configuración. Con tan pocas
filas no se puede concluir qué características ayudan a esta clase, y es la
clase que limita el F1-macro. `sqli` tiene 11 filas de prueba después de
eliminar duplicados, así que su recall también es solo indicativo.

### 8.3 Flujos de una sola dirección

En el 100% de las filas del laboratorio el conteo de paquetes de vuelta
(backward) es 0 y el origen del flujo es el servidor. Falta el lado del
cliente de todas las conversaciones del laboratorio: no están representadas
las peticiones del atacante ni las cargas de inyección SQL. Otro extractor
(NTLFlowLyzer) produce flujos bidireccionales a partir de la misma captura,
lo que apunta a una falla de la extracción y no de la captura. Esa
comparación se verificó en un solo PCAP (`{ntl_pcap}`), no en las 12 capturas
del laboratorio; para las otras 11 se asume, no se comprobó. Todas las características de
vuelta son constantes para las filas del laboratorio, y las clases del
laboratorio se separan únicamente por lo que envió el servidor.

### 8.4 Duplicados

Con las columnas de puertos presentes, los archivos originales tienen 94 (A)
y 100 (B) filas duplicadas exactas. Una vez que se quitan los puertos,
{dups_A} (A) y {dups_B} (B) filas son idénticas a otra fila y se eliminaron,
conservando una copia. El efecto está concentrado: `PortScan` pasa de 25,409
filas a 554 y `sqli` de 179 a 58 (A) y 57 (B). Por lo tanto, las métricas
describen el desempeño por patrón de flujo distinto, no por flujo capturado,
y las frecuencias de clase ya no reflejan el tráfico capturado
(`decisions.md`, decisión 4). Las filas casi idénticas no se eliminaron y
pueden caer a ambos lados de la división entre entrenamiento y prueba, lo que
puede volver optimistas los puntajes de prueba.

### 8.5 Otras limitaciones

- **Chi-cuadrado y características dispersas.** Con bins por cuantiles, una
  característica en la que un solo valor cubre cerca del 90% de las filas o
  más queda reducida a un único bin y recibe un puntaje de 0, sin importar la
  información que contenga. Esto les ocurre a {single_bin_A} características
  no constantes en A y a {single_bin_B} en B, en su mayoría contadores de
  banderas. En B solo 39 características sobreviven a Pearson, así que las 30
  mejores abarcan la mayoría e incluyen una característica con puntaje 0
  (`Bwd Packet Length Min`).
- **Un solo modelo.** La comparación usa un RandomForest. Los modelos
  sensibles a las características redundantes pueden beneficiarse más de los
  filtros.
- **Cinco folds.** El error estándar que usa la regla de selección es, a su
  vez, una estimación aproximada.
- **A y B comparten sus filas externas,** así que los resultados de los dos
  conjuntos de datos no son réplicas independientes.

## 9. Reproducibilidad

Ejecutar en este orden con `.venv/bin/python`:

| Script | Etapa |
|---|---|
| `src/00_setup.py` | Carga y verificación de los archivos originales |
| `src/01_audit.py` | Auditoría de calidad |
| `src/02_clean.py`, `src/02b_duplicates_check.py` | Limpieza y verificación de duplicados |
| `src/03_split.py`, `src/03b_handoff.py` | División en entrenamiento y prueba, y paquete de datos |
| `src/04_filters.py` | Los tres filtros sobre el conjunto de entrenamiento completo |
| `src/05_evaluate.py`, `src/05b_plots.py` | Evaluación y gráficas |
| `src/06_deliverables.py`, `src/06b_report.py` | Listas finales y este informe |
| `src/06c_report_pdf.py`, `src/06d_report_docx.py` | Exportación de este informe a PDF y Word |

Los parámetros están en `src/config.py`. Las semillas son fijas. Las etapas 1
a 4 y la 6 se ejecutaron dos veces y produjeron archivos idénticos; la etapa
5 (cerca de una hora) se ejecutó una vez.

## Anexo. Matrices de confusión de las demás configuraciones evaluadas en prueba

Las matrices de las listas finales están en la sección 7. Aquí se incluyen
las del baseline y las de la configuración seleccionada en las demás
familias. Los títulos de las gráficas están en inglés.

{appendix}

## Glosario

| Término | Significado |
|---|---|
| ANOVA F | Prueba estadística que mide cuánto difiere el promedio de una característica entre las clases. Aquí se usa para decidir cuál característica de un par correlacionado se conserva. |
| Atajo (shortcut) | Característica que permite acertar la etiqueta por una particularidad del montaje de los datos y no por el comportamiento del ataque. |
| Baseline | Modelo de referencia, entrenado con todas las características, contra el que se compara cada subconjunto. |
| Bin | Tramo en el que se agrupan los valores de una característica continua para convertirla en categorías. |
| Característica (feature) | Columna de entrada del modelo; cada una describe una propiedad de un flujo. |
| Chi-cuadrado | Estadístico que mide la asociación entre dos variables categóricas a partir de una tabla de conteos. |
| `class_weight="balanced"` | Opción de scikit-learn que da más peso a las clases con menos filas. |
| Cuantil | Valor que divide los datos ordenados en partes con la misma cantidad de filas. |
| Error estándar | Medida de cuánto varía un promedio de una repetición a otra; aquí, entre los folds. |
| Estratificado | División que mantiene en cada parte la misma proporción de cada clase. |
| F1-macro | Promedio del F1 (media armónica de precisión y recall) de todas las clases, con el mismo peso para cada clase. |
| Flujo | Resumen estadístico de los paquetes intercambiados en una conexión de red. |
| Fold | Cada una de las particiones de la validación cruzada. |
| Fuga de información (leakage) | Uso, durante el ajuste, de información de las filas con las que después se evalúa. |
| Low Variance | Filtro que elimina las características que casi no cambian de valor. |
| Matriz de confusión | Tabla que cruza la clase real de cada fila con la clase que predijo el modelo. |
| MCC | Coeficiente de correlación de Matthews; resume toda la matriz de confusión en un solo número y es robusto al desbalance de clases. |
| Pipeline | Secuencia de pasos que se aplican uno después del otro. |
| PR-AUC macro | Área bajo la curva de precisión y recall, promediada entre las clases. |
| RandomForest | Modelo de clasificación que combina muchos árboles de decisión. |
| Recall | Proporción de las filas de una clase que el modelo identifica correctamente. |
| Test (prueba) | Filas reservadas para la evaluación final, que no intervienen en ningún ajuste. |
| Top k | Las k características con mayor puntaje. |
| Train (entrenamiento) | Filas con las que se ajustan los filtros y se entrena el modelo. |
| V de Cramér | Versión del Chi-cuadrado llevada a una escala común, que corrige el efecto del tamaño de la muestra y de la cantidad de bins. |
| Validación cruzada | Procedimiento que divide el entrenamiento en varias partes y evalúa en cada una un modelo entrenado con las demás. |
"""

# Table headers and short phrases used by the table builders in 06b_report.py.
TEXT_ES = {
    "cv_header": ["Configuración", "A caract.", "A F1-macro", "A PR-AUC",
                  "A MCC", "B caract.", "B F1-macro", "B PR-AUC", "B MCC"],
    "test_header": ["Conjunto de datos", "Configuración", "Características",
                    "F1-macro", "PR-AUC macro", "MCC",
                    "Tiempo de entrenamiento (s)"],
    "recall_header": ["Etiqueta", "A filas de prueba", "A baseline", "A final",
                      "B filas de prueba", "B baseline", "B final"],
    "feature_header": ["Posición", "Característica", "Chi-cuadrado",
                       "V de Cramér", "En la otra lista"],
    "selected": " **(seleccionada)**",
    "yes": "sí",
    "no": "no",
    "or": " o ",
    "range": "{low} a {high}",
    "worst": "{rows} filas `{true}` predichas como `{predicted}`",
    "appendix_alt": "Matriz de confusión en prueba, conjunto de datos "
                    "{dataset}, {config}",
}
