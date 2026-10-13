# Selección de características con métodos de filtro para un conjunto de datos de flujos de un IDS

Proyecto "Arquitectura IDS impulsada por IA". Autora: Laura Camila Herrera
Murillo. Curso: [NOMBRE DEL CURSO]. Profesor: [NOMBRE]. Fecha: 2026-10-14.

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
  **20 características para A** y **30 para B** (de 77) y obtienen un
  F1-macro (promedio del F1 de todas las clases, con el mismo peso para cada
  una) de 0.9678 y 0.9701 en las filas de prueba (test),
  frente a 0.9713 y 0.9669 con todas las características. Las
  diferencias son menores que el ruido entre folds (particiones de la
  validación cruzada) medido en la validación cruzada (error estándar de
  aproximadamente 0.0072).
- El pipeline combinado (secuencia de pasos: Low Variance, luego Pearson,
  luego Chi-cuadrado) alcanza al baseline (modelo de referencia con todas
  las características) con menos características que Chi-cuadrado solo
  (20 frente a 40 en A, 30 frente a 40 en B), porque Chi-cuadrado solo
  llena los primeros puestos con características redundantes.
- **Los puntajes no deben leerse como capacidad de detectar ataques.** Las
  características identifican casi a la perfección de cuál de las dos fuentes
  de datos proviene una fila, y las etiquetas que nombran el mismo ataque en
  las dos fuentes nunca se confunden entre sí. La sección 8 explica esta y
  las demás limitaciones.

## 2. Datos y preparación

Cada conjunto de datos une filas de un conjunto externo (IDS2025) con flujos
capturados en un laboratorio virtual, extraídos con `cicflowmeter`: 79
características y la variable objetivo `newLabel` con 12 etiquetas. A y B se
procesan por separado con el mismo código.

| Paso | A | B |
|---|---|---|
| Filas originales | 370,545 | 345,192 |
| Duplicados exactos eliminados (después de quitar las columnas de puertos) | 35,062 | 36,079 |
| Filas limpias | 335,314 | 308,940 |
| Filas de entrenamiento (estratificado, 20% para prueba, semilla 42) | 268,251 | 247,152 |
| Filas de prueba | 67,063 | 61,788 |

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
elimina 9 características en ambos conjuntos de datos: las 5
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
cada una se corta en 10 bins (tramos) por cuantiles, con límites
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

- Modelo: `RandomForestClassifier` con 200 árboles y
  `class_weight="balanced"` (pesos de clase inversos a su frecuencia),
  idéntico para todos los subconjuntos y sin ajuste de hiperparámetros.
- Métricas: F1-macro, PR-AUC macro (área bajo la curva de precisión y
  recall, promediada entre clases), MCC (coeficiente de correlación de
  Matthews, que usa toda la matriz de confusión) y recall (proporción de
  filas de una clase que el modelo detecta) por clase. La exactitud
  (accuracy) no se reporta como métrica principal porque las dos clases más
  grandes reúnen cerca del 70% de las filas.
- Las configuraciones se comparan con validación cruzada estratificada de
  5 folds dentro del entrenamiento; los filtros se vuelven a ajustar en
  cada fold.
- Regla de selección, fijada antes de ver los resultados: dentro de cada
  familia, la configuración con menos características cuyo F1-macro promedio
  esté a un error estándar o menos de la mejor.
- Las filas de prueba se usan una sola vez, para el baseline y para la
  configuración seleccionada de cada familia.

## 5. Comparación

### 5.1 Validación cruzada dentro del entrenamiento (todas las configuraciones)

Promedio de 5 folds; el F1-macro se muestra con su error estándar.
`combined_pX_kY` significa umbral de Pearson X y las Y mejores por
Chi-cuadrado.

| Configuración | A caract. | A F1-macro | A PR-AUC | A MCC | B caract. | B F1-macro | B PR-AUC | B MCC |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 77 | 0.9618 ± 0.0072 **(seleccionada)** | 0.9789 | 0.9938 | 77 | 0.9675 ± 0.0081 **(seleccionada)** | 0.9830 | 0.9928 |
| `variance_qc_0.999` | 68 | 0.9621 ± 0.0074 **(seleccionada)** | 0.9794 | 0.9939 | 68 | 0.9677 ± 0.0081 **(seleccionada)** | 0.9824 | 0.9930 |
| `pearson_0.90` | 45 | 0.9630 ± 0.0075 **(seleccionada)** | 0.9792 | 0.9936 | 46 | 0.9574 ± 0.0076 **(seleccionada)** | 0.9753 | 0.9932 |
| `pearson_0.95` | 52 | 0.9613 ± 0.0077 | 0.9759 | 0.9942 | 51 | 0.9612 ± 0.0081 | 0.9827 | 0.9934 |
| `pearson_0.99` | 61 | 0.9633 ± 0.0073 | 0.9789 | 0.9940 | 61 | 0.9611 ± 0.0076 | 0.9805 | 0.9929 |
| `chi2_top_10` | 10 | 0.8627 ± 0.0091 | 0.8903 | 0.9618 | 10 | 0.8707 ± 0.0048 | 0.8823 | 0.9633 |
| `cramers_v_top_10` | 10 | 0.4396 ± 0.0056 | 0.5127 | 0.2685 | 10 | 0.4198 ± 0.0104 | 0.4976 | 0.2802 |
| `chi2_top_20` | 20 | 0.9250 ± 0.0079 | 0.9574 | 0.9849 | 20 | 0.9254 ± 0.0056 | 0.9555 | 0.9831 |
| `cramers_v_top_20` | 20 | 0.9046 ± 0.0050 | 0.9178 | 0.9731 | 20 | 0.9022 ± 0.0037 | 0.9139 | 0.9726 |
| `chi2_top_30` | 30 | 0.9253 ± 0.0081 | 0.9577 | 0.9862 | 30 | 0.9242 ± 0.0057 | 0.9555 | 0.9845 |
| `cramers_v_top_30` | 30 | 0.9475 ± 0.0078 | 0.9759 | 0.9856 | 30 | 0.9606 ± 0.0048 **(seleccionada)** | 0.9718 | 0.9896 |
| `chi2_top_40` | 40 | 0.9594 ± 0.0078 **(seleccionada)** | 0.9710 | 0.9940 | 40 | 0.9576 ± 0.0080 **(seleccionada)** | 0.9717 | 0.9928 |
| `cramers_v_top_40` | 40 | 0.9682 ± 0.0075 **(seleccionada)** | 0.9800 | 0.9943 | 40 | 0.9618 ± 0.0085 | 0.9803 | 0.9928 |
| `combined_p0.90_k10` | 10 | 0.9266 ± 0.0067 | 0.9513 | 0.9839 | 10 | 0.9198 ± 0.0058 | 0.9514 | 0.9828 |
| `combined_p0.90_k20` | 20 | 0.9613 ± 0.0067 **(seleccionada)** | 0.9687 | 0.9937 | 20 | 0.9548 ± 0.0077 | 0.9673 | 0.9932 |
| `combined_p0.90_k30` | 30 | 0.9662 ± 0.0077 | 0.9820 | 0.9936 | 30 | 0.9676 ± 0.0080 **(seleccionada)** | 0.9789 | 0.9933 |
| `combined_p0.95_k10` | 10 | 0.9218 ± 0.0073 | 0.9539 | 0.9846 | 10 | 0.9225 ± 0.0050 | 0.9544 | 0.9814 |
| `combined_p0.95_k20` | 20 | 0.9244 ± 0.0073 | 0.9580 | 0.9859 | 20 | 0.9533 ± 0.0060 | 0.9702 | 0.9910 |
| `combined_p0.95_k30` | 30 | 0.9586 ± 0.0076 | 0.9705 | 0.9943 | 30 | 0.9672 ± 0.0083 | 0.9812 | 0.9932 |
| `combined_p0.99_k10` | 10 | 0.9072 ± 0.0078 | 0.9570 | 0.9807 | 10 | 0.8974 ± 0.0061 | 0.9354 | 0.9753 |
| `combined_p0.99_k20` | 20 | 0.9275 ± 0.0075 | 0.9592 | 0.9860 | 20 | 0.9232 ± 0.0038 | 0.9554 | 0.9832 |
| `combined_p0.99_k30` | 30 | 0.9614 ± 0.0069 | 0.9687 | 0.9942 | 30 | 0.9580 ± 0.0082 | 0.9734 | 0.9929 |

![Comparación en validación cruzada, conjunto de datos A](figures/cv_comparison_A.png)
![Comparación en validación cruzada, conjunto de datos B](figures/cv_comparison_B.png)

### 5.2 Prueba (baseline y la configuración seleccionada de cada familia)

| Conjunto de datos | Configuración | Características | F1-macro | PR-AUC macro | MCC | Tiempo de entrenamiento (s) |
|---|---|---|---|---|---|---|
| A | `baseline` | 77 | 0.9713 | 0.9688 | 0.9939 | 19.0 |
| A | `variance_qc_0.999` | 68 | 0.9721 | 0.9685 | 0.9941 | 21.6 |
| A | `pearson_0.90` | 45 | 0.9727 | 0.9685 | 0.9938 | 14.3 |
| A | `chi2_top_40` | 40 | 0.9659 | 0.9575 | 0.9940 | 16.0 |
| A | `cramers_v_top_40` | 40 | 0.9715 | 0.9691 | 0.9944 | 15.0 |
| A | `combined_p0.90_k20` | 20 | 0.9678 | 0.9650 | 0.9939 | 13.7 |
| B | `baseline` | 77 | 0.9669 | 0.9682 | 0.9927 | 19.9 |
| B | `variance_qc_0.999` | 68 | 0.9683 | 0.9656 | 0.9928 | 19.8 |
| B | `pearson_0.90` | 46 | 0.9692 | 0.9676 | 0.9932 | 15.5 |
| B | `cramers_v_top_30` | 30 | 0.9540 | 0.9586 | 0.9892 | 9.5 |
| B | `chi2_top_40` | 40 | 0.9621 | 0.9674 | 0.9927 | 19.9 |
| B | `combined_p0.90_k30` | 30 | 0.9701 | 0.9661 | 0.9932 | 15.7 |

### 5.3 Recall por clase en prueba, baseline frente a la lista final

| Etiqueta | A filas de prueba | A baseline | A final | B filas de prueba | B baseline | B final |
|---|---|---|---|---|---|---|
| Botnet ARES | 275 | 0.971 | 0.982 | 275 | 0.971 | 0.989 |
| Brute Force | 1378 | 1.000 | 1.000 | 1378 | 0.999 | 1.000 |
| Dos/DDos | 4273 | 0.997 | 0.998 | 4273 | 0.998 | 0.998 |
| Infiltration | 6 | 0.667 | 0.667 | 6 | 0.667 | 0.667 |
| PortScan | 111 | 0.982 | 0.982 | 111 | 0.973 | 0.982 |
| Web Attack | 406 | 0.993 | 0.993 | 406 | 0.993 | 0.993 |
| benign | 20834 | 0.988 | 0.990 | 20560 | 0.988 | 0.989 |
| bruteforce | 13344 | 1.000 | 1.000 | 8422 | 1.000 | 1.000 |
| dos_httpflood | 25403 | 0.999 | 0.998 | 24706 | 0.999 | 0.999 |
| dos_slowhttptest | 165 | 1.000 | 1.000 | 164 | 1.000 | 1.000 |
| recon | 857 | 0.995 | 0.988 | 1476 | 0.997 | 0.997 |
| sqli | 11 | 1.000 | 1.000 | 11 | 1.000 | 1.000 |

Lectura de los resultados:

- Low Variance y Pearson eliminan entre 9 y 32 características sin un cambio
  consistente: la mayor diferencia con el baseline en validación cruzada es
  de 0.010 en F1-macro (Pearson 0.90 en B, error estándar 0.008) y no aparece
  en prueba. Las características redundantes no perjudicaban al baseline: un
  conjunto de árboles las tolera.
- Chi-cuadrado solo necesita 40 características para alcanzar al baseline;
  con 10 o 20 es claramente peor.
- El ordenamiento por la V de Cramér no es una mejor alternativa: con 10
  características su F1-macro cae a 0.44 (A), 0.42 (B) y no detecta ninguna fila
  de `sqli`.
- El mayor error individual del baseline es 200 filas `benign` predichas como `Dos/DDos` en A y 203 filas `benign` predichas como `Dos/DDos`
  en B.

## 6. Por qué el pipeline combinado supera a Chi-cuadrado solo

Chi-cuadrado puntúa cada característica por separado. Dos características que
son copia una de la otra reciben el mismo puntaje alto y ambas entran en las
k mejores, aunque la segunda no aporta nada. Evidencia, para la misma
cantidad de características:

| | A (k = 20) | B (k = 30) |
|---|---|---|
| F1-macro en validación cruzada, Chi-cuadrado top k | 0.9250 | 0.9242 |
| F1-macro en validación cruzada, combinado top k | 0.9613 | 0.9676 |
| Pares de características con correlación absoluta mayor que 0.90 dentro de Chi-cuadrado top k | 21 | 34 |
| Lo mismo, dentro de la lista combinada | 0 | 0 |
| Características presentes tanto en Chi-cuadrado top k como en la lista combinada | 6 | 13 |

Los primeros puestos del ordenamiento por Chi-cuadrado están ocupados por
grupos como `Total Length of Fwd Packets` y `Subflow Fwd Bytes`, o
`Fwd Packet Length Mean` y `Avg Fwd Segment Size`, que son la misma magnitud
con dos nombres. El pipeline combinado deja que Pearson conserve primero una
característica por grupo, de modo que cada uno de los k puestos corresponde a
información distinta.

## 7. Listas finales de características

Elegidas por la regla de selección: `combined_p0.90_k20` para A y `combined_p0.90_k30` para B.
La regla no se ajustó después de ver los resultados (`decisions.md`,
decisión 8), y por eso las dos listas tienen longitudes distintas. Las listas
se ajustaron con el conjunto de entrenamiento completo de cada conjunto de
datos y están guardadas en `results/final_features_A.csv` y
`results/final_features_B.csv`.

### Conjunto de datos A (20 características)

| Posición | Característica | Chi-cuadrado | V de Cramér | En la otra lista |
|---|---|---|---|---|
| 1 | `Total Length of Fwd Packets` | 684,568 | 0.565 | sí |
| 2 | `Fwd Packet Length Mean` | 643,034 | 0.632 | sí |
| 3 | `Max Packet Length` | 635,350 | 0.628 | sí |
| 4 | `Fwd IAT Mean` | 552,886 | 0.479 | sí |
| 5 | `Fwd IAT Total` | 498,573 | 0.454 | no |
| 6 | `Flow Packetss` | 495,073 | 0.453 | sí |
| 7 | `Fwd Header Length` | 484,335 | 0.549 | sí |
| 8 | `Total Fwd Packets` | 483,129 | 0.548 | sí |
| 9 | `Flow Bytess` | 469,214 | 0.441 | sí |
| 10 | `Packet Length Mean` | 468,722 | 0.591 | no |
| 11 | `Idle Max` | 267,288 | 0.499 | no |
| 12 | `Idle Mean` | 266,483 | 0.498 | sí |
| 13 | `Idle Std` | 240,515 | 0.473 | sí |
| 14 | `Active Max` | 232,852 | 0.466 | no |
| 15 | `Active Min` | 206,749 | 0.439 | sí |
| 16 | `Flow IAT Min` | 198,537 | 0.430 | sí |
| 17 | `Init_Win_bytes_backward` | 198,103 | 0.608 | sí |
| 18 | `Fwd IAT Min` | 181,074 | 0.411 | sí |
| 19 | `Init_Win_bytes_forward` | 175,271 | 0.572 | sí |
| 20 | `Min Packet Length` | 158,262 | 0.768 | sí |

### Conjunto de datos B (30 características)

| Posición | Característica | Chi-cuadrado | V de Cramér | En la otra lista |
|---|---|---|---|---|
| 1 | `Total Length of Fwd Packets` | 602,592 | 0.552 | sí |
| 2 | `Average Packet Size` | 568,494 | 0.619 | no |
| 3 | `Fwd Packet Length Mean` | 494,064 | 0.632 | sí |
| 4 | `Idle Mean` | 488,277 | 0.497 | sí |
| 5 | `Max Packet Length` | 478,543 | 0.622 | sí |
| 6 | `Flow Packetss` | 452,778 | 0.451 | sí |
| 7 | `Flow IAT Mean` | 451,824 | 0.451 | no |
| 8 | `Flow Duration` | 445,525 | 0.448 | no |
| 9 | `Fwd IAT Mean` | 429,245 | 0.439 | sí |
| 10 | `Flow Bytess` | 411,823 | 0.430 | sí |
| 11 | `Idle Std` | 364,752 | 0.430 | sí |
| 12 | `Fwd Header Length` | 336,172 | 0.522 | sí |
| 13 | `Total Fwd Packets` | 288,740 | 0.540 | sí |
| 14 | `Active Min` | 261,817 | 0.515 | sí |
| 15 | `Active Mean` | 243,260 | 0.496 | no |
| 16 | `Init_Win_bytes_backward` | 181,329 | 0.606 | sí |
| 17 | `Flow IAT Min` | 174,425 | 0.376 | sí |
| 18 | `Init_Win_bytes_forward` | 174,182 | 0.594 | sí |
| 19 | `Min Packet Length` | 143,281 | 0.761 | sí |
| 20 | `Fwd Packet Length Min` | 143,193 | 0.761 | no |
| 21 | `Bwd IAT Total` | 142,165 | 0.758 | no |
| 22 | `Bwd IAT Mean` | 141,991 | 0.758 | no |
| 23 | `Subflow Bwd Bytes` | 124,241 | 0.709 | no |
| 24 | `Bwd IAT Min` | 118,020 | 0.691 | no |
| 25 | `Bwd Packet Length Mean` | 106,954 | 0.658 | no |
| 26 | `Fwd IAT Min` | 91,347 | 0.304 | sí |
| 27 | `Bwd Packetss` | 83,502 | 0.581 | no |
| 28 | `Down Up Ratio` | 66,174 | 0.517 | no |
| 29 | `Protocol` | 61 | 0.016 | no |
| 30 | `Bwd Packet Length Min` | 0 | 0.000 | no |

### A frente a B

- Compartidas (16): `Active Min`, `Flow Bytess`, `Flow IAT Min`, `Flow Packetss`, `Fwd Header Length`, `Fwd IAT Mean`, `Fwd IAT Min`, `Fwd Packet Length Mean`, `Idle Mean`, `Idle Std`, `Init_Win_bytes_backward`, `Init_Win_bytes_forward`, `Max Packet Length`, `Min Packet Length`, `Total Fwd Packets`, `Total Length of Fwd Packets`
- Solo en A (4): `Active Max`, `Fwd IAT Total`, `Idle Max`, `Packet Length Mean`
- Solo en B (14): `Active Mean`, `Average Packet Size`, `Bwd IAT Mean`, `Bwd IAT Min`, `Bwd IAT Total`, `Bwd Packet Length Mean`, `Bwd Packet Length Min`, `Bwd Packetss`, `Down Up Ratio`, `Flow Duration`, `Flow IAT Mean`, `Fwd Packet Length Min`, `Protocol`, `Subflow Bwd Bytes`

![Matriz de confusión en prueba, conjunto de datos A, lista final](figures/confusion_A_combined_p0.90_k20.png)

![Matriz de confusión en prueba, conjunto de datos B, lista final](figures/confusion_B_combined_p0.90_k30.png)

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

- **Confusiones entre etiquetas equivalentes:** 0 en total, en
  las 12 configuraciones evaluadas en prueba, en cualquiera de
  los dos sentidos. Si las características describieran el ataque en sí, los
  flujos del mismo tipo de ataque de las dos fuentes se confundirían al menos
  ocasionalmente.
- **Prueba de la fuente:** se entrenó un RandomForest con la misma
  configuración para responder únicamente "¿externo o laboratorio?", usando
  solo filas `benign`, la única etiqueta presente en las dos fuentes. Con las
  listas finales clasifica mal 0 de 20,834 filas de
  prueba en A y 2 de 20,560 en B.

Parte de la razón se ve en las características: 27 de las 77
características valen cero en todas las filas del laboratorio (ver 8.3), así
que cualquier valor distinto de cero marca una fila externa. La lista final
de A contiene 1 de ellas (`Init_Win_bytes_backward`) y la lista final de
B contiene 9 (`Init_Win_bytes_backward`, `Bwd IAT Total`, `Bwd IAT Mean`, `Subflow Bwd Bytes`, `Bwd IAT Min`, `Bwd Packet Length Mean`, `Bwd Packetss`, `Down Up Ratio`, `Bwd Packet Length Min`).

Consecuencia: el modelo separa las fuentes y después las clases dentro de
cada fuente. Los puntajes altos no demuestran que las características
seleccionadas caractericen un tipo de ataque, y no dicen nada sobre el
tráfico de otra red. El experimento que lo pondría a prueba, entrenar con una
fuente y evaluar con las etiquetas equivalentes de la otra, no se realizó.

### 8.2 Infiltration

`Infiltration` tiene 29 filas en total: 23 en entrenamiento y
6 en prueba. En prueba el modelo detecta 4 de
6 en A y 4 de 6 en B, en todas las
configuraciones; una fila equivale a 17 puntos de recall. En validación
cruzada, sobre las 23 filas de entrenamiento, el recall va de
0.57 a 0.70 en A y de 0.52 a 0.70 en B según la configuración. Con tan pocas
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
comparación se verificó en un solo PCAP (`fw_benign_20260828_01`), no en las 12 capturas
del laboratorio; para las otras 11 se asume, no se comprobó. Todas las características de
vuelta son constantes para las filas del laboratorio, y las clases del
laboratorio se separan únicamente por lo que envió el servidor.

### 8.4 Duplicados

Con las columnas de puertos presentes, los archivos originales tienen 94 (A)
y 100 (B) filas duplicadas exactas. Una vez que se quitan los puertos,
35,062 (A) y 36,079 (B) filas son idénticas a otra fila y se eliminaron,
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
  información que contenga. Esto les ocurre a 17 características
  no constantes en A y a 16 en B, en su mayoría contadores de
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

Los parámetros están en `src/config.py`. Las semillas son fijas. Las etapas 1
a 4 y la 6 se ejecutaron dos veces y produjeron archivos idénticos; la etapa
5 (cerca de una hora) se ejecutó una vez.

## Anexo. Matrices de confusión de las demás configuraciones evaluadas en prueba

Las matrices de las listas finales están en la sección 7. Aquí se incluyen
las del baseline y las de la configuración seleccionada en las demás
familias. Los títulos de las gráficas están en inglés.

![Matriz de confusión en prueba, conjunto de datos A, baseline](figures/confusion_A_baseline.png)
![Matriz de confusión en prueba, conjunto de datos B, baseline](figures/confusion_B_baseline.png)

![Matriz de confusión en prueba, conjunto de datos A, variance_qc_0.999](figures/confusion_A_variance_qc_0.999.png)
![Matriz de confusión en prueba, conjunto de datos B, variance_qc_0.999](figures/confusion_B_variance_qc_0.999.png)

![Matriz de confusión en prueba, conjunto de datos A, pearson_0.90](figures/confusion_A_pearson_0.90.png)
![Matriz de confusión en prueba, conjunto de datos B, pearson_0.90](figures/confusion_B_pearson_0.90.png)

![Matriz de confusión en prueba, conjunto de datos A, chi2_top_40](figures/confusion_A_chi2_top_40.png)
![Matriz de confusión en prueba, conjunto de datos B, chi2_top_40](figures/confusion_B_chi2_top_40.png)

![Matriz de confusión en prueba, conjunto de datos A, cramers_v_top_40](figures/confusion_A_cramers_v_top_40.png)
![Matriz de confusión en prueba, conjunto de datos B, cramers_v_top_30](figures/confusion_B_cramers_v_top_30.png)

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
