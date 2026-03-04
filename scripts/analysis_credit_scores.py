# %% [markdown]
# # 📊 Credit Scoring Analytics – Banking Credit Risk Analysis
# ---
# Fecha de Creación : Octubre 2022
#
# - **Notebook** by Julio César Martínez I.
# - **Supported** by Francisco Alfaro & Alfonso Tobar
# - **Code Reviewer** Oscar Flores

# %% [markdown]
# # Licencia
# ---
#
# Copyright @2023 by Julio César Martínez Izaguirre
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License

# %% [markdown]
# # Tabla de Contenido
# ---
#
# 1. Propósito del Proyecto
# 2. Etapa 1 : Carga de Datos
# 3. Etapa 2 : Transformación de Datos
# 4. Etapa 3 : Trabajando valores ausentes
# 5. Etapa 4 : Clasificación de datos
# 6. Etapa 5 : Comprobación de hipótesis
# 7. Conclusión General

# %% [markdown]
# # Introducción
# ---
#
# Este proyecto consiste en **preparar un informe** para la división de préstamos de un banco. Deberemos averiguar si el estado civil y el número de hijos de un cliente tienen un impacto en el incumplimiento de pago de un préstamo. El banco ya tiene algunos datos sobre la solvencia crediticia de los clientes.
#
# Nuestro informe se tendrá en cuenta al crear una **puntuación de crédito** para un cliente potencial. La **puntuación de crédito** se utiliza para evaluar la capacidad de un prestatario potencial para pagar su préstamo.

# %% [markdown]
# # Propósito del Proyecto
# ---
#
# Dar a conocer si un cliente puede ser acreedor o no a una línea de crédito por parte del banco de acuerdo a una serie de características personales y profesionales registradas por la entidad financiera.
#
# **Hipótesis**
#
# 1. Averiguar si existe alguna conexión entre tener hijos y pagar un prestamo a tiempo.
# 2. Determinar si existe una conexión entre el estado civil y el pago a tiempo de un préstamo.
# 3. Saber si existe alguna conexión con el nivel de ingresos y el pago de un préstamo.
# 4. Cómo afectan los diferentes propósitos del préstamo al reembolso a tiempo del préstamo.

# %% [markdown]
# **Descripción de los datos**
#
# Estas son las columnas que hay en nuestro dataset y una breve descripción de ellas.
#
# - `children` - el número de hijos en la familia
# - `days_employed` - experiencia laboral en días
# - `dob_years` - la edad del cliente en años
# - `education` - la educación del cliente
# - `education_id` - identificador de educación
# - `family_status` - estado civil
# - `family_status_id` - identificador de estado civil
# - `gender` - género del cliente
# - `income_type` - tipo de empleo
# - `debt` - ¿había alguna deuda en el pago de un préstamo?
# - `total_income` - ingreso mensual
# - `purpose` - el propósito de obtener un préstamo

# %% [markdown]
# ## Etapa 1 : Carga de Archivos y Exploración de Datos
#
# Vamos a comenzar con lo básico, vamos a cargar las librerías que vamos a utilizar para este proyecto y enseguida haremos una breve exploración de datos inicial para conocer lo que hay en nuestro set.
#
# **Carga de Librerias**

# %%
import pandas as pd
import numpy as np

pd.options.display.max_rows = 100

# %% [markdown]
# **Lectura de Datos**

# %%
dataclients = pd.read_csv('/content/credit_scoring_eng.csv')

# %% [markdown]
# **Exploración de Datos**

# %%
dataclients.info()

# %%
dataclients.head()

# %% [markdown]
# > Podemos ver que existen valores ausentes `NaN` en las columnas `days_employed` y `total_income`. Además, encontramos algunos valores negativos en columnas que vamos a investigar más adelante, por lo pronto analicemos un poco más los valores ausentes que hemos encontrado.

# %% [markdown]
# **Examinando valores ausentes**

# %%
dataclients[dataclients.isna().any(axis=1)]

# %% [markdown]
# > Los **valores ausentes** parecen estar relacionados entre ambas columnas, es decir, son simétricos. Para estar seguros de esto, vamos a realizar un recuento de los valores ausentes en ambas columnas.

# %%
dataclients[['days_employed', 'total_income']].isna().sum()

# %% [markdown]
# > El número de filas de la tabla filtrada coincide con el número de valores ausentes, esto quiere decir que las personas que no tienen experiencia laboral tampoco registran un ingreso mensual.
#
# **Supuesto**
#
# Lo anterior probablemente tenga una relación con el nivel de educación ya que una educación deficiente o de bajo nivel tiene como consecuencias desempleo y/o empleos no formales, para ello, vamos a investigar a los clientes que no tienen datos sobre la característica identificada y la columna con los valores ausentes.

# %% [markdown]
# **Análisis de Educación**

# %%
dataclients['education'].unique()

# %% [markdown]
# > Podemos ver datos duplicados que contienen el mismo grado escolar pero escritos en mayúsculas y minúsculas, vamos arreglar un poco esto para tener un resultado más preciso.

# %%
dataclients['education'] = dataclients['education'].str.lower()
dataclients['education'].unique()

# %% [markdown]
# **Filtrando Tabla**
#
# Una vez limpios nuestros datos haremos un filtro para conocer los valores ausentes en el nivel de educación secundaria

# %%
dataclients[dataclients['education'] ==
            'secondary education']['days_employed'].value_counts(dropna=False)

# %% [markdown]
# > Nuestros datos nos indican que existen un total de **13,694** registros de nivel de **educación secundaria**, de los cuales hay un total de 1,540 datos sobre los valores ausentes, ahora, comprobemos la distribución entre los datos ausentes de nivel secundaria y el total de clientes.

# %%
nan_data = 2174
nan_sec_edu = 1540

total_dis = nan_sec_edu / nan_data
f'El porcentaje de distribución del bajo nivel educativo es de: {(total_dis):.0%}'

# %% [markdown]
# **Posibles razones por las que hay valores ausentes en los datos**
#
# El resultado nos muestra que un 71% de los valores ausentes proviene de clientes que tienen un bajo nivel educativo. Esto se acerca más a nuestra teoría de que sean clientes desempleados y/o tengan empleos informales.
#
# Sin embargo esto no cubre el 10% de los valores ausentes por lo cuál no podemos definir si se trata de un patrón exacto o son valores aleatorios. Vamos a comprobar si nuestros valores ausentes son al azar o hay un patrón con esto.
#

# %%
# Comprobando la distribución en el conjunto de datos entero
totaldata = len(dataclients)
nan_data = 2174

distribution = nan_data / totaldata
f'La distribución del conjunto de datos entero es: {(distribution):.0%}'

# %% [markdown]
# > Podemos ver que la distribución de ambos conjuntos es muy diferente, esto significa que no se cubre el 100% de los valores ausentes en nuestra tabla filtrada lo cual quiere decir que, aunque se aproxima, no existe una relación entre los valores ausentes y el nivel de educación.
#
# **Supuesto No 2**
#
# Vamos a tomar otra columna para conocer si hay alguna otra relación con ella o descartarla. Ahora tomaremos la columna de `income_type`. Se piensa que el tipo de empleo puede ser otro factor por el cual un cliente no tendría experiencia laboral y tampoco reciba ingresos. Por ejemplo algunos que estén retirados o se encuentren desempleados, vamos a explorar un poco en ello.

# %% [markdown]
# **Revisando distribución de clientes por ingreso**

# %%
dataclients['income_type'].isna().sum()

# %%
dataclients['income_type'].value_counts(dropna=False)

# %% [markdown]
# **Analizando valores ausentes**
#
# Hagamos un filtro entre el tipo de empleo y los clientes cuyo estatus actual se encuentran retiradas

# %%
dataclients[(dataclients['days_employed'].isna()) &
            (dataclients['income_type'] == 'retiree')]

# %% [markdown]
# > Con estos datos nos damos cuenta de que solo 413 personas están retiradas y esto justificaría la falta de días de experiencia laboral; sin embargo en el resto de valores los clientes cuentan con un empleo en la actualidad, lo cual no justifica la falta de estos valores, vamos a ver si existe algún otro valor de desempleo para conocer esta relación, para ello, tomaremos la columna `employee` en esta ocasión.

# %%
dataclients[(dataclients['days_employed'].isna()) &
            (dataclients['income_type'] == 'employee')]

# %% [markdown]
# > Podemos ver que existen solo 1,105 datos de los 2,174 ausentes. Podemos concluir entonces que es posible que haya ocurrido un error al ingresar los datos o que los clientes no hayan querido responder a estos valores por cuestiones de tipo emocional ya sea miedo o timidez pero no podemos saberlo porque no existe algún otro patrón en específico que nos lo indíque. En general no existen patrones que nos puedan indicar la aparición de estos valores ausentes.

# %% [markdown]
# **Comparación**
#
# A continuación vamos a crear una visión general de dos dataframes, una con los datos nulos y otra sin ellos para poder visualizar sus distribuciones.

# %%
# DataFrame con valores ausentes
dataclients.describe()

# %%
# DataFrame sin valores ausentes
dataclients.dropna().describe()

# %% [markdown]
# **Conclusión**
#
# Podemos observar con certeza que en ambos dataframes son prácticamente iguales y no existe una distribución especial para alguna de nuestras variables lo cual nos lleva a la conclusión de que nuestros datos nulos se generaron al azar.

# %% [markdown]
# ## Etapa 2: Transformación de datos
#
# Durante esta etapa vamos a repasar cada columna para ver qué problemas podemos encontrar en ellas, comenzaremos con la eliminación de valores duplicados y continuaremos con la corrección de la información educativa en caso de ser necesario.

# %% [markdown]
# **Revisando Columna: Education**

# %%
dataclients['education'].unique()

# %% [markdown]
# > Los valores se arreglaron previamente por lo que no será necesario tratar esta columna.

# %% [markdown]
# **Columna: Children**
#
# Distribución en la columna

# %%
dataclients['children'].value_counts(dropna=False)

# %% [markdown]
# > Notamos que en la columna de **Children** hay valores negativos, es probable que se trate de un error de entrada en los datos o quizás al importar el dataframe, no puede existir un valor de **-1 hijos**. Calcularemos primero el porcentaje que representan estos datos respecto al resto del conjunto y después cambiaremos su **valor en positivo**. También tenemos un registro de 76 clientes con 20 hijos, algo que es muy improbable, lo más seguro es que hayan anotado un cero de más y se trate de 2 hijos, además se trata de un valor muy bajo que no impactará en nuestro análisis si lo agregamos a la columna con 2 hijos.

# %% [markdown]
# **Distribución de negativos**

# %%
total_data = 21525
negative_children = 47

negative_children_percentage = negative_children/total_data
f'El porcentaje de valores negativos en la columna children es: {(negative_children_percentage):.0%}'

# %% [markdown]
# Con esta operación podemos ver que el número de datos negativos no es relevante para nuestro análisis y podrían quedarse así; sin embargo esto no es correcto y cambiaremos los valores negativos a positivos.

# %%
# Imprimiendo valores
dataclients['children'].unique()

# %%
# Corrección de valores / verificando cambios
dataclients['children'] = dataclients['children'].abs()
dataclients['children'] = dataclients['children'].replace(20, 2)
dataclients['children'].unique()

# %% [markdown]
# **Columna: Days Employed**
#
# Revisando distribución de valores

# %%
dataclients['days_employed'].value_counts()

# %%
dataclients['days_employed'].unique()

# %% [markdown]
#
# En esta columna encontramos los siguientes problemas:
#
# 1. Valores NaN
# 2. Valores negativos.
# 3. Valores de coma flotante.
#
# Estos problemas pueden deberse a errores en la manipulación de la información o quizás al factor humano al ingresar los datos al dataframe. Sea cual sea la causa debemos solucionarlo.
#
# Vamos abordar estos problemas de la siguiente manera:
#
# 1. Los valores ausentes los colocaremos en cero.
# 2. Los valores negativos los cambiaremos a positivos.
# 3. Los decimales los pasaremos a enteros pues es difícil interpretar 35.2 días.

# %%
dataclients['days_employed'] = dataclients['days_employed'].fillna(0)
dataclients['days_employed'] = dataclients['days_employed'].abs()
dataclients['days_employed'] = dataclients['days_employed'].astype(int)

# %% [markdown]
# **Comprobando correcciones**

# %%
dataclients['days_employed'].value_counts()

# %% [markdown]
# Al aplicar 'value_counts' podemos ver que existen valores muy grandes, quizás estén expresados en otras unidades, vamos a intentar convertirlos en horas.

# %%
dataclients['days_employed'].max()

# %%
dataclients['days_employed'] = round(dataclients['days_employed'] / 24)
dataclients['days_employed'] = dataclients['days_employed'].astype(int)
dataclients['days_employed'].max()

# %% [markdown]
# Ahora el valor máximo es de 16740 días que equivalen a 45 años, ahora nuestros datos tienen más sentido.

# %% [markdown]
# **Columna: Dob Years**
#
# Ahora echemos un vistazo a la edad de clientes para ver si hay algún problema allí.

# %%
dataclients['dob_years'].unique()

# %% [markdown]
# Análisis de distribución

# %%
dataclients['dob_years'].value_counts()

# %% [markdown]
# Existe un total de 101 registros en cero (0), esto no puede ser posible debido a que un cliente no podría tener cero años, de ser ese el caso estaríamos hablando de un recién nacido. Esta cantidad corresponde al 0.4% del total de los datos lo cuál no impacta en nuestro análisis.
#
# Vamos a reemplazar estos valores por la media.

# %%
m = dataclients['dob_years'].mean()
print('El valor promedio de la columna dob years es:', m)

# %%
dataclients['dob_years'].replace(0, 43, inplace=True)
dataclients['dob_years'].value_counts()

# %% [markdown]
# **Columna: Family_Status**

# %%
print('Valores únicos:', dataclients['family_status'].unique())
dataclients['family_status'].value_counts(dropna=False)

# %% [markdown]
# No se aprecian valores extraños relevantes para nuestro análisis.

# %% [markdown]
# **Columna: Gender**
#
# Vamos a revisar ahora la columna 'gender' para observar si hay problemas en ella.

# %%
print('valores únicos:', dataclients['gender'].unique())
dataclients['gender'].value_counts()

# %% [markdown]
# Encontramos un valor extraño llamado 'XNA', como no se trata de un valor representativo podemos integrarlo a cualquiera de los dos géneros sin que genere un impacto en nuestro análisis, en mi caso lo integraré al género F debido a la frecuencia que tiene la columna.

# %%
dataclients['gender'] = dataclients['gender'].replace('XNA', 'F')
dataclients['gender'].value_counts()

# %% [markdown]
# **Columna: Income Type**
#
# Vamos a revisar la columna 'income_type'.

# %%
print('valores únicos:', dataclients['income_type'].unique())
dataclients['income_type'].value_counts(dropna=False)

# %% [markdown]
# No se aprecian valores extraños o relevantes para nuestro análisis.

# %% [markdown]
# **DataFrame: Datos Duplicados**
#
# Revisemos si existen datos duplicados.

# %%
print(
    'Total de datos duplicados:',
    dataclients.duplicated().sum())

# %%
# Abordando duplicados
dataclients = dataclients.drop_duplicates().reset_index(drop=True)
dataclients.duplicated().sum()

# %% [markdown]
# Comprobando tamaño del dataframe después de manipulaciones

# %%
dataclients.shape

# %% [markdown]
# **Conclusión**
#
# Después de limpiar algunas columnas podemos ver un ajuste en el número de valores, aún nos falta por revisar los valores ausentes de la columna 'total_income' pero es algo que resolveremos en la siguiente etapa.

# %% [markdown]
# ## Etapa 3: Trabajar con valores ausentes

# %% [markdown]
# Vamos a trabajar con algunos valores ausentes que encontramos en la columna **total_income**, pero primero vamos a revisar los valores de **índice y valor** en education y family status para visualizar un poco la correlación que existe entre estas variables

# %%
dataclients[['education_id', 'education']].value_counts()

# %%
dataclients[['family_status_id', 'family_status']].value_counts()

# %%
dataclients['dob_years'].value_counts().sort_values()

# %% [markdown]
# ### Restaurar valores ausentes en `total_income`

# %% [markdown]
#
# Para restaurar valores ausentes en la columna vamos a crear una columna nueva de categorías por edad dentro del dataframe, la utilizaré más adelante como referencia para tratar los valores ausentes.
#

# %%


def age_range(row):
    if row['dob_years'] < 20:
        return '10-20'
    elif row['dob_years'] < 30:
        return '20-30'
    elif row['dob_years'] < 40:
        return '30-40'
    elif row['dob_years'] < 50:
        return '40-50'
    elif row['dob_years'] < 60:
        return '50-60'
    else:
        return '60+'


# %%
dataclients.apply(age_range, axis=1)

# %% [markdown]
# **Realizando Clasificación por Edades**

# %%
dataclients['age_range'] = dataclients.apply(age_range, axis=1)
dataclients.head()

# %% [markdown]
# > Uno de los factores que nos puede ayudar a rellenar los valores ausentes en la columna de ingresos es el rango de edad y para ello utilizaremos nuestra columna con el rango de edades, para ello, crearemos una tabla que solo tenga datos sin valores ausentes, estos datos se utilizarán para restaurar los valores ausentes.

# %%
data_ref = dataclients.dropna()
data_ref.head()

# %% [markdown]
# **Examinando promedio y mediana**
#
# Vamos analizar los promedios de los ingresos en función de los factores que clasificamos.

# %%
mean = data_ref.groupby('age_range')['total_income'].mean()
print('Promedio de ingresos por rango de edad:', mean)

# %%


# %%
median = data_ref.groupby('age_range')['total_income'].median()
print('Mediana de ingresos por rango de edad:', median)

# %%


# %% [markdown]
# Vamos a utilizar el promedio para sustituir valores ausentes en el ingreso de los clientes, para ello, vamos a desarrollar una función que nos ayude a sustituir los NaN por el promedio que corresponde a la clasificación de rangos por edad.
#

# %%
def missing_replace(dataclients):
    mean = dataclients.groupby('age_range')['total_income'].mean()

    dataclients.loc[(dataclients['age_range'] == '10-20') &
                    dataclients['total_income'].isna(), 'total_income'] = mean.loc['10-20']
    dataclients.loc[(dataclients['age_range'] == '20-30') &
                    dataclients['total_income'].isna(), 'total_income'] = mean.loc['20-30']
    dataclients.loc[(dataclients['age_range'] == '30-40') &
                    dataclients['total_income'].isna(), 'total_income'] = mean.loc['30-40']
    dataclients.loc[(dataclients['age_range'] == '40-50') &
                    dataclients['total_income'].isna(), 'total_income'] = mean.loc['40-50']
    dataclients.loc[(dataclients['age_range'] == '50-60') &
                    dataclients['total_income'].isna(), 'total_income'] = mean.loc['50-60']
    dataclients.loc[(dataclients['age_range'] == '60+') &
                    dataclients['total_income'].isna(), 'total_income'] = mean.loc['60+']

    return dataclients


# %%
# Comprobando función
missing_replace(dataclients)

# %% [markdown]
# **Sustituyendo Valores Ausentes**

# %%
dataclients = missing_replace(dataclients)
ti_nan = dataclients['total_income'].isna().sum()
print('Valores ausentes en total income:', ti_nan)
dataclients.head()

# %% [markdown]
# ###  Restaurar valores en `days_employed`

# %% [markdown]
#
# Ahora vamos a rellenar los valores ausentes en función del tipo de empleo pues tiene sentido que la experiencia laboral este en función de esta columna, para ello, buscaremos los datos de la media, la mediana y después, determinaremos cuales datos serán mejores para sustituirlos.

# %%
# Contar Ceros
dataclients['days_employed'].value_counts()

# %%
# Distribución de las medianas de `days_employed` en función de los parámetros identificados
gender_median = dataclients.groupby('income_type')['days_employed'].median()
gender_median

# %%
# Distribución de las medias de `days_employed` en función de los parámetros identificados
gender_mean = dataclients.groupby(
    'income_type')['days_employed'].mean().round()
gender_mean

# %% [markdown]
# Recordemos que nuestros datos nulos los transformamos en valores en ceros, por lo tanto, ahora vamos a sustituirlos por el valor de la mediana, para ello, vamos a desarrollar otra función que nos ayude a realizar esta sustitución.

# %%


def replace_days_employed(dataclients):
    gender_median = dataclients.groupby(
        'income_type')['days_employed'].median()

    dataclients.loc[(dataclients['income_type'] == 'business') & (
        dataclients['days_employed'] == 0), 'days_employed'] = gender_median.loc['business']
    dataclients.loc[(dataclients['income_type'] == 'civil servant') & (
        dataclients['days_employed'] == 0), 'days_employed'] = gender_median.loc['civil servant']
    dataclients.loc[(dataclients['income_type'] == 'employee') & (
        dataclients['days_employed'] == 0), 'days_employed'] = gender_median.loc['employee']
    dataclients.loc[(dataclients['income_type'] == 'paternity / maternity leave') & (
        dataclients['days_employed'] == 0), 'days_employed'] = gender_median.loc['paternity / maternity leave']
    dataclients.loc[(dataclients['income_type'] == 'retiree') & (
        dataclients['days_employed'] == 0), 'days_employed'] = gender_median.loc['retiree']
    dataclients.loc[(dataclients['income_type'] == 'student') & (
        dataclients['days_employed'] == 0), 'days_employed'] = gender_median.loc['student']
    dataclients.loc[(dataclients['income_type'] == 'unemployed') & (
        dataclients['days_employed'] == 0), 'days_employed'] = gender_median.loc['unemployed']

    return dataclients


# %%
# Comprobando la función
replace_days_employed(dataclients)

# %% [markdown]
# **Sustituyendo valores en days employed**

# %%
# Aplicar la función al income_type
dataclients = replace_days_employed(dataclients)
dataclients['days_employed'].value_counts()

# %%
# Conteo de NaN
dataclients['days_employed'].isna().sum()

# %% [markdown]
# **Comprobando NaN en el Dataset**

# %%
dataclients.isna().sum()

# %% [markdown]
# ## Etapa 4 : Clasificación de datos
#
# Para poder responder a las preguntas y probar las diferentes hipótesis vamos a trabajar en esta sección con datos clasificados. A continuación vamos a clasificar los datos en función de las siguientes columnas:
#
# 1. Cantidad de Hijos
# 2. Incumplimiento de Pagos
# 3. Estado Civil
# 4. Ingreso Mensual
#
# Esto para comprobar o rechazar las hipótesis que se establecieron desde el inicio.
#

# %%
# Valores de los datos seleccionados para la clasificación
dataclients['children'].value_counts()

# %% [markdown]
# En este punto vamos a clasificar la columna 'debt' creando una nueva columna llamada 'debt_name' en la cual, para fines prácticos, se colocarán las siguientes anotaciones: no deudor = 0 y deudor = 1, para lograrlo, desarrollaremos una función que nos permita clasificar los datos.

# %%
# Contando valores
dataclients['debt'].value_counts()

# %%
# Creamos una función para nuestra nueva columna


def debt_name(row):
    if row['debt'] == 0:
        return 'no deudor'
    else:
        return 'deudor'


# %%
# Comprobamos que funcione
dataclients.apply(debt_name, axis=1)

# %% [markdown]
# **Creando clasificación en Debt**

# %%
dataclients['debt_name'] = dataclients.apply(debt_name, axis=1)
dataclients.head()

# %% [markdown]
#

# %% [markdown]
# Revisamos el resto de variables que vamos a utilizar

# %%
dataclients['family_status'].value_counts()

# %%
dataclients['total_income'].value_counts()

# %% [markdown]
# Vamos a comprobar los valores únicos

# %%
# Comprobar los valores únicos
dataclients['children'].unique()

# %%
dataclients['debt'].unique()

# %%
dataclients['family_status'].unique()

# %%
dataclients['total_income'].unique()

# %% [markdown]
#
#
# > Bloque con sangría
#
#
# La siguiente clasificación será si tienen hijos y el incumplimiento de un pago, para ello, vamos a crear una nueva columna con esta clasificación de hijos que usaremos más adelante para conocer estos datos.
#

# %%
# Función para clasificar los datos
children_ref = dataclients[['children', 'debt']]


def children_range(row):
    if row['children'] == 0:
        return 'sin hijos'
    else:
        return 'con hijos'


# %%
# Creando columna con las categorías y cuenta los valores en ellas
dataclients['children_range'] = dataclients.apply(children_range, axis=1)

# %%
# Revisando todos los datos numéricos en la columna seleccionada para la clasificación
dataclients['children_range'].value_counts()

# %%
# Estadísticas resumidas para la columna
dataclients['children_range'].describe()

# %% [markdown]
# **Clasificación del ingreso mensual**

# %%
# Estadísticas resumidas
dataclients['total_income'].describe()

# %% [markdown]
# Vamos a crear la función utilizando rangos en multiplos de 3, la razón de esto es porque tomamos como referencia el dataframe con la función describe y sus valores.

# %%
# Función para clasificar en diferentes grupos numéricos basándose en rangos


def income_range(row):
    if row['total_income'] < 3000:
        return '0 - 3000'
    elif row['total_income'] < 9000:
        return '3000 - 9000'
    elif row['total_income'] < 15000:
        return '9000 - 15000'
    elif row['total_income'] < 21000:
        return '15000 - 21000'
    elif row['total_income'] < 27000:
        return '21000 - 27000'
    else:
        return '27000+'

# %% [markdown]
# Creando columna nueva con categorías de ingresos


# %%
dataclients['income_range'] = dataclients.apply(income_range, axis=1)

# %%
# Distribución de categorías
dataclients['income_range'].value_counts().sort_values()

# %% [markdown]
# **Vamos a clasificar la columna de 'purpose'** <a id="back"> </a>

# %%
# Función para clasificar los datos


def change_purpose(row):
    purpose_category = 'unknow'

    if 'wedding' in row['purpose']:
        purpose_category = 'wedding'

    elif ('real' in row['purpose']) or ('state' in row['purpose']) or ('house' in row['purpose']) or ('property' in row['purpose']) or ('housing' in row['purpose']):
        purpose_category = 'real estate'

    elif 'car' in row['purpose']:
        purpose_category = 'car'

    elif ('education' in row['purpose']) or ('educated' in row['purpose']) or ('university' in row['purpose']):
        purpose_category = 'education'

    return purpose_category


# %%
# Comprobamos la función
dataclients.apply(change_purpose, axis=1).value_counts()

# %% [markdown]
# Creando nueva clasificación y verificamos

# %%
dataclients['purpose_classified'] = dataclients.apply(change_purpose, axis=1)
dataclients.head()

# %% [markdown]
# ## Etapa 5: Comprobación de las hipótesis
#

# %% [markdown]
# Una vez que tenemos todo nuestro set de datos clasificado ahora vamos a comprobar nuestras hipótesis.
#
# **HIPÓTESIS No1 : ¿EXISTE UNA CORRELACIÓN ENTRE TENER HIJOS Y PAGAR A TIEMPO?**

# %%
dataclients.groupby(['children', 'debt_name'])['debt'].value_counts()

# %%
# Contamos el total de hijos
dataclients['children'].value_counts()

# %%
# Calculando la tasa de incumplimiento en función del número de hijos
tasa_childrens = {'children': [0, 1, 2, 3, 4, 5],
                  'deudor': [(1063/14091)*100, (445/4855)*100, (202/2128)*100, (27/330)*100, (4/41)*100, (0/9)*100],
                  'no deudor': [(13028/14091)*100, (4410/4855)*100, (1926/2128)*100, (303/330)*100, (37/41)*100, (9/9)*100]
                  }

# %%
pd.DataFrame(tasa_childrens)

# %% [markdown]
# **Conclusión**
#
# Gracias a nuestra investigación podemos ver el **9.7%** de los clientes con más deuda son aquellos que tienen 4 hijos respecto de quienes no tienen hijos que representan solo el **7.5%** de incumplimiento y al mismo tiempo tienen mayor tasa de cumplimiento.
#
# Podríamos hablar de los clientes que tienen **5 hijos** pero solo se trata de 9 clientes con estas características por lo que no es una cantidad considerable dentro de nuestro dataframe.
#
# Hay que poner más énfasis en aquellos que tienen entre **1 y 2 hijos** pues representan una gran cantidad de clientes y las tasas de incumplimiento también son altas llegando al **9.5%** y **9.2%** respectivamente.

# %% [markdown]
# **HIPÓTESIS NO2 : ¿EXISTE UNA CORRELACIÓN ENTRE LA SITUACIÓN FAMILIAR Y EL PAGO A TIEMPO?**

# %%
# Comprobando los datos del estado familiar y los pagos a tiempo
# Total de deudores por familia y la suma de deudores en c/u
debt_group = dataclients.groupby(
    'family_status').agg({'debt': ['size', 'sum']})
debt_group

# %%
# Ahora queremos obtener la tasa de los pagos a tiempo (no deudores)
# Vamos a restar el grupo de deudores del total para obtener a los No Deudores
# Luego vamos a dividir y sacar el porcentaje para conocer la tasa de los pagos a tiempo.

cumplimiento = debt_group['debt']['size'] - debt_group['debt']['sum']
tasa_pagos_puntuales = cumplimiento / debt_group['debt']['size']
print('Las tasas de pagos puntuales por estado civil')
print('.............................')

tasa_pagos_puntuales*100

# %%
# Calcular la tasa de incumplimiento basada en el estado familiar
# Ya teniamos el total y la suma de deudores previamente
# Hacemos la operación correspondiente para obtener la tasa de incumplimiento

tasa_incumplimiento = debt_group['debt']['sum'] / debt_group['debt']['size']
print('Las Tasas de incumplimiento por estado civil')
print('.............................')

tasa_incumplimiento * 100

# %% [markdown]
# **Conclusión**
#
# En nuestra tabla de **incumplimiento** podemos ver que los **viudos** representan la menor tasa de incumplimiento respecto a los que viven en **unión libre** que representan hasta el **9.3%**. Por otra parte los **casados**, quienes representan la mayor cantidad de nuestro dataframe, representan el **7.5%** de incumplimiento.

# %% [markdown]
# **HIPÓTESIS: ¿EXISTE UNA CORRELACIÓN ENTRE EL NIVEL DE INGRESOS Y EL PAGO A TIEMPO?**

# %%
# ComprOBANDO los datos del nivel de ingresos y los pagos a tiempo
income_ref = dataclients.groupby(['income_range', 'debt_name'])[
    'debt'].value_counts()
income_ref

# %%
# Contamos el total de ingresos
income_total = dataclients.groupby('income_range').size()
income_total

# %%
# Distribución
income_conversion = (income_ref / income_total)*100
income_conversion

# %% [markdown]
# **Conclusión**
#
# La mayor **tasa de cumplimiento** la tienen aquellos clientes cuyos ingresos mensuales rondan los **3000 a 9000**. Por otra parte los clientes con más **incumplimiento** son aquellos que sus ingresos mensuales rondan entre los **21000 y 27000**. Es muy probable que esto se deba a que realmente son pocos los clientes con el nivel de ingreso de 3 a 9 mil mientras que la cantidad de clientes va a más del doble en aquellos que ganan entre 21 mil y los 27 mil.

# %% [markdown]
# **¿Cómo afecta el propósito del crédito a la tasa de incumplimiento?**

# %%
# Consultando los porcentajes de tasa de incumplimiento para cada propósito del crédito
debt_grouped = dataclients.groupby(['purpose_classified', 'debt_name'])[
    'debt'].value_counts()

# %%
# Contamos el total de la columna 'purpose'
purpose = dataclients.groupby('purpose_classified').size()

# %%
# Calculamos la distribución
purpose_conversion = (debt_grouped / purpose)*100
purpose_conversion

# %% [markdown]
# **Conclusión**
#
# En lo que respecta al propósito del crédito podemos ver que la tasa de deudores más alta, del 9%, pertenece a los clientes que han buscado invertir en un auto, en general, podemos ver que el propósito de un crédito no afecta en el cumplimiento o incumplimiento de este.
#

# %% [markdown]
# # Conclusión general
#
# Hemos pasado un buen rato analizando datos e información clave y hemos encontrado diversas problematicas en todo el ejercicio como son:
#
# 1. Hemos resuelto problemas de valores ausentes en las columnas de la experiencia laboral y el ingreso mensual.
# 2. Hemos tenido que resolver problemas adicionales repasando cada columna.
# 3. Hemos hecho un análsis previo para encontrar algún patrón o factor común que tuviera relación con los valores ausentes de las columnas de experiencia laboral e ingreso mensual.
# 4. Abordamos el tema de valores ausentes en la columna de ingreso mensual creando clasificaciones por rango de edades y calculando la media y la mediana determinando cuál era la más apropiada forma de aplicar al conjunto de datos.
# 5. Abordamos el tema de los duplicados dentro de nuestro dataset.
# 6. Encontramos valores negativos y algunos valores improbables en las columnas de número de hijos, días de experiencia laboral y en ingreso mensual. Estos valores solo se restauraron y se ordenaron.
# 7. Se crearon nuevas clasificaciones de rango de edad, rango de ingreso mensual, rango de hijos y nombres para las columnas de deudores con el objetivo de analizar, responder y comprobar nuestras hiótesis.
#
# **Hipótesis**
#
# La primera hipótesis sobre la relación entre el número de hijos y el pago de un préstamo es aceptada.
# La segunda hipótesis sobre la relación entre el estado civil y el pago de un préstamo es aceptada.
# La tercera hipótesis sobre el nivel de ingreso y el pago de una deuda es aceptada.
# La cuarta hipótesis sobre cómo afectan las razones para pedir un préstamo y el incumplimiento aceptada pues la relación que existe es baja.
#
# **Conclusión**
#
# Después de este análisis determinamos que los aspectos más relevantes para solicitar un crédito son el número de hijos, el estado civil y el nivel de ingreso mensual. Mientras que las razones por las cuales éste se solicita no tienen mucha influencia en el cumplimiento del crédito.
