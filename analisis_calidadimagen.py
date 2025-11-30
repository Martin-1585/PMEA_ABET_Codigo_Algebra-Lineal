import numpy as np
import matplotlib.pyplot as pit
import sqlite3

NAME_DB = "calidad_imagenes.db"

# Manejo de la base de datos
def conectar_db():
    return sqlite3.connect(NAME_DB)
# Funcion para crear la tabla de datos con la que se trabajara
def inicializar_sistema():
    """Inicializa la base de datos si no existe"""
    # Se instancia una vairable connect para que verifique si existe una archivo con el nombre de la variable global
    connect = conectar_db()
    cursor = connect.cursor() # Crear un cursor para ejecutar comandos SQL
    cursor.execute('''CREATE TABLE IF NOT EXISTS datos_experimentales (id INTEGER PRIMARY KEY, compresion REAL, calidad REAL)''')
    # Si el archivo con los datos experiementales no existe se crea un archivo .db y se inicializa con datos de ejemplo
    cursor.execute('SELECT count(*) FROM datos_experimentales') # Verificar la cantidad de registros en la tabla
    if cursor.fetchone()[0] == 0: # Si no hay datos en el archivo, se inicliza con datos proporcionados enla descripción del problema
        print("\tBase de datos inicializada")
        data = [(10,92), (20, 88), (30, 83), (40, 78), (50, 73), (60, 68)] # Datos experimentales iniciales
        cursor.executemany('INSERT INTO datos_experimentales (compresion, calidad) VALUES (?, ?)', data) # Insertar los datos experimentales en el archivo .db
        connect.commit() # Guardar los cambios en la base de datos
    connect.close() # Cerrar la conexión a la base de datos
    
def agregar_datos():
    print("\tAgregar datos a la base de datos")
    try:
        compresion_input = input("Ingrese el nivel de compresión(%): ") #Ingresar un nivel de compresión para realizar el cálculo de calidad mediante la factorización por QR en polinimios cuadráticos
        compresion = float(compresion_input) # Transformar la entrada en un valor numérico de tipo float
        x_hist, y_hist = obtener_datos() # Obtener losa datos de compresión y calidad para tener los datos históricos
        c = calcular_modeloqr(x_hist, y_hist) # Calcular los coeficientes del modelo cuadrático mediante la factorización por QR
        quility_obtained = c[0] + c[1] * compresion + c[2] * (compresion ** 2) # Usar los coeficientes obtenidos para lograr calcular la calidad de la compressión ingresada
        print(f"Calidad de imagen estimada: {quility_obtained:.2f}%") # Mostrar la calidad obtenida con dos decimales
        connect = conectar_db() # Conectar a la base de datos
        cursor = connect.cursor() # Crear un cursos para ejecutar comandos SQL
        cursor.execute('INSERT INTO datos_experimentales (compresion, calidad) VALUES (?, ?)',(compresion, quility_obtained)) # Insertar los nuevos datos en la base de datos
        connect.commit() # Guardar los cambios realizados a la base de datos
        connect.close() # Cerrar la conexión a la base de datos
    except ValueError: # Capturar errores de conversión de tipos
        print("Entrada inválida. Por favor, ingrese un número válido.")
        
def obtener_datos():
    connect = conectar_db() # Conectar a la base de datos
    cursor = connect.cursor() #Crear un cursos para ejecutar comandos SQL
    cursor.execute('SELECT compresion, calidad FROM datos_experimentales ORDER BY compresion ASC') # Ordenar los datos de compresión de manera ascendente
    data = cursor.fetchall() # Obtener todos los registros de la tabla
    connect.close() # Cerrar la conexión a la base de datos
    # Extraer los datos de compresión y calidad en arreglos numpy
    x = np.array([row[0] for row in data], dtype= float) # Extraer los datos de compresión y transformalos en tipo float
    y = np.array([row[1] for row in data], dtype= float) # Extraer los datos de calidad y transformalos en tipo float
    return x, y

def calcular_modeloqr(x, y):
    col_1 = np.ones(len(x)) # Crear una columna de unos para el término independiente
    A = np.column_stack((col_1, x, x**2)) # Crear la matriz de diseño para un modelo cuadrático
    Q, R = np.linalg.qr(A) # Factorización QR de la matriz A
    d = np.dot(Q.T, y) # Obtener el termino superior triangular del sistema
    c = np.linalg.solve(R, d) # Resolver el sistema triangular para obtener los coeficientes del modelo
    return c

def modo_prediccion():
    x_train, y_train = obtener_datos() # Obtener los datos históricos de compresión y calidad
    
    if(len(x_train) < 3): # Si no hay sufiencientes datos de compreción se lanza una excepción
        print("No hay suficientes datos para un modelo cuadrático (3 min)")
        return
    
    c = calcular_modeloqr(x_train, y_train) # Devolver los coeficientes del modelo cuadrático mediante la factorización por QR
    print(f"Modelo cuadrático: Calidad = {c[0]:.2f} + {c[1]:.2f}*Compresión + {c[2]:.4f}*Compresión^2") # Mostrar el modelo cuadrático obtenido con dos decimales para los términos lineales y cuatro para el término cuadrático
    
    while True:
        val_input = input("Ingrese el nivel de compresión(%) para calcular la calidad o 'salir' para terminar: ") # Solicitar al usuario un nivel de compresión, que posteriormente será graficado para representar su curva
        if val_input.lower() == 'salir': # Si el usuario ingresa 'salir', se termina el modo de predicción
            break
        try:
            val_x = float(val_input) # Se transforma la entrada en un valor numérico de tipo float
            val_y = c[0] + c[1] * val_x + c[2] * (val_x ** 2) # Se calcula la calidad estimada usando el modelo cuadrático obtenido
            # Verficación de estado y veracidad del calculo realizado
            interpolacion = min(x_train) <= val_x <= max(x_train) # Verificar si el valor ingresado está dentro del rango de datos históricos (10 - 60)
            estado = "CONFIABLE" if interpolacion else "PRECAUCION" # Determinar si la predicción es confiable o puede estar trabajando con valores tan dispersos que pueden poco acertados
            color = "green" if interpolacion else "red" # Asignar un color verde para predicciones confiables y rojo para predicciones con precaución
            print(f"Calidad estimada: {val_y:.2f}% - Estado: {estado}") # Mostrar la calidad estimada y el estado del valor obtenido
            # Graficar
            pit.figure(figsize=(8,5)) # Establecer el tamaño de la figura
            pit.scatter(x_train, y_train, color = "black", label = "Datos Historicos") # Graficar los datos históricos de compresión y calidad
            
            rango = np.linspace(min(min(x_train), val_x) - 5, max(max(x_train), val_x) + 5, 100) # Determinar el rango de datos con los que se va a trabajar en el grafico
            y_graphic = c[0] + c[1] * rango + c[2] * (rango ** 2) # Calcular los valores de calidad para el rango de calidad
            
            pit.plot(rango, y_graphic, color = "purple", alpha = 0.5, label = "Modelo Cuadrático") # Graficar la curva del modelo cuadrático
            pit.scatter([val_x], [val_y], color = color, s = 150, marker= '*', label = "Predicción Actual") # Personalizar el marcador del punto de predicción actual
            
            pit.title(f"Comprension: {val_x} -> Calidad: {val_y:.2f}% \nEstado: {estado}") # Título del gráfico con la información de la predicción actual acerca de la calidad calculada
            pit.xlabel("Nivel de Compresión (%)") # Eje x
            pit.ylabel("Calidad de Imagen") # Eje y
            pit.legend(); pit.grid(True, alpha=0.3) # Leyenda y cuadrícula
            print("Cierra la gráfica para continuar...")
            pit.show() # Mostrar el gráfico
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número válido.")
            
def main():
    inicializar_sistema()
    while True:
        print("\n" + "="*44)
        print("| Análisis de Calidad de Imágenes          |")
        print("| 1. Agregar datos de compresión y calidad |")
        print("| 2. Modo predicción de calidad            |")
        print("| 3. Ver datos actuales                    |")
        print("| 4. SALIR                                 |")
        print("="*44)
        
        op = input("Seleccione una opción (1-4): ")
        if op == '1':
            agregar_datos()
        elif op == '2':
            modo_prediccion()
        elif op == '3':
            x, y = obtener_datos()
            print(f"\nDatos {len(x)} registros: \n{np.column_stack((x, y))}") # Mostrar los datos actuales en la base de datos
        elif op == '4':
            print("\tSALIENDO DEL PROGRAMA...")
            break
        
if __name__ == "__main__":
    main()