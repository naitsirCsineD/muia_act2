import os
import random
import cv2
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")
import numpy as np


def seleccionar_y_ecualizar_imagen_cv2(directorio):
    """
    Selecciona una imagen al azar de un directorio, la ecualiza y la muestra.
    Compatible con imágenes de 16 bits usando OpenCV.
    
    :param directorio: Ruta del directorio que contiene las imágenes.
    """
    if not os.path.exists(directorio):
        print(f"El directorio '{directorio}' no existe.")
        return
    
    # Filtrar archivos de imagen
    archivos = [archivo for archivo in os.listdir(directorio) if archivo.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]
    
    if not archivos:
        print("No se encontraron imágenes en el directorio.")
        return
    
    # Seleccionar una imagen al azar
    imagen_seleccionada = random.choice(archivos)
    ruta_imagen = os.path.join(directorio, imagen_seleccionada)
    
    try:
        # Cargar la imagen
        print(f"Procesando imagen: {imagen_seleccionada}")
        img = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)  # Lee la imagen con su profundidad original
        
        if img is None:
            print("No se pudo cargar la imagen.")
            return
        
        # Verificar si es de 16 bits
        if img.dtype == 'uint16':
            print("La imagen es de 16 bits. Normalizando y convirtiendo a 8 bits para ecualizar...")
            img = (img / 256).astype('uint8')  # Normalizar a 8 bits
        
        # Convertir a escala de grises si es necesario
        if len(img.shape) == 3:
            print("La imagen es en color. Convirtiendo a escala de grises...")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ecualizar el histograma
        img_ecualizada = cv2.equalizeHist(img)
        
        # Mostrar la imagen original y la ecualizada
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Imagen Original")
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Imagen Ecualizada")
        plt.imshow(img_ecualizada, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        aplicar_kernels(img)
        
    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen: {e}")




def aplicar_kernels(imagen_path):
    """
    Aplica diferentes kernels a una imagen y muestra los resultados.
    
    :param imagen_path: Ruta de la imagen que será procesada.
    """
    # Cargar la imagen en escala de grises
    img = imagen_path
    if img is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        return

    # Kernel promedio (Blur)
    kernel_promedio = cv2.blur(img, (5, 5))

    # Kernel Gaussiano
    kernel_gaussiano = cv2.GaussianBlur(img, (5, 5), 0)

    # Kernel de Mediana
    kernel_mediana = cv2.medianBlur(img, 5)

    # Kernel de detección de bordes verticales
    kernel_borde_vertical = cv2.filter2D(img, -1, np.array([[-1, 0, 1],
                                                            [-2, 0, 2],
                                                            [-1, 0, 1]]))

    # Kernel de detección de bordes horizontales
    kernel_borde_horizontal = cv2.filter2D(img, -1, np.array([[-1, -2, -1],
                                                              [ 0,  0,  0],
                                                              [ 1,  2,  1]]))
    

    kernel_realce_nitidez=cv2.filter2D(img,-1,np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]]))
    
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    realce_local=unsharp


    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Bordes horizontales
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Bordes verticales
    sobel = cv2.magnitude(sobel_x, sobel_y)
    Bordes_Sobel=cv2.convertScaleAbs(sobel)

    threshold1=10
    threshold2=20
    Borde_Canny=cv2.Canny(img, threshold1, threshold2)


    #Separa elementos


    kernel_size=10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ErosMorf=cv2.erode(img, kernel, iterations=1)

    #Potencia y aumenta los contornos, haciendo que lod detalles se magnifiquen
    #capaz de fusuonar objetos separados

    kernel_size=10
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    dilatacion=cv2.dilate(img, kernel, iterations=1)

    # Útil para separa objetos, elimina detalles en función del elemento estructural
    # Se eliminan detalles, y se magnifica lo que queda
    kernel_size=10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    apertura=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


    # consolidar formas y rellenar pixeles en una imagen
    kernel_size=10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cierre=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Calcular el gradiente morfológico (dilatación - erosión)
    gradient_image = cv2.subtract(dilatacion, ErosMorf)


    #Detección de textos cualquiero tipo de detalle
    topHat=cv2.subtract(img, dilatacion)



    # Mostrar las imágenes procesadas
    plt.figure(figsize=(15, 10))

    plt.subplot(4, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(4, 4, 2), plt.imshow(kernel_promedio, cmap='gray'), plt.title('Kernel Promedio')
    plt.axis('off')

    plt.subplot(4, 4, 3), plt.imshow(kernel_gaussiano, cmap='gray'), plt.title('Kernel Gaussiano')
    plt.axis('off')

    plt.subplot(4, 4, 4), plt.imshow(kernel_mediana, cmap='gray'), plt.title('Kernel Mediana')
    plt.axis('off')

    plt.subplot(4, 4, 5), plt.imshow(kernel_borde_vertical, cmap='gray'), plt.title('Borde Vertical')
    plt.axis('off')

    plt.subplot(4, 4, 6), plt.imshow(kernel_borde_horizontal, cmap='gray'), plt.title('Borde Horizontal')
    plt.axis('off')


    plt.subplot(4, 4, 7), plt.imshow(kernel_realce_nitidez, cmap='gray'), plt.title('Realce de Nitidez')
    plt.axis('off')

    plt.subplot(4, 4, 8), plt.imshow(realce_local, cmap='gray'), plt.title('Realce Local')
    plt.axis('off')

    plt.subplot(4, 4, 9), plt.imshow(Bordes_Sobel, cmap='gray'), plt.title('Bordes con Sobel')
    plt.axis('off')

    plt.subplot(4, 4, 10), plt.imshow(Borde_Canny, cmap='gray'), plt.title('Bordes con Canny')
    plt.axis('off')

    plt.subplot(4, 4, 11), plt.imshow(ErosMorf, cmap='gray'), plt.title('Erosión Mofológica')
    plt.axis('off')

    plt.subplot(4, 4, 12), plt.imshow(dilatacion, cmap='gray'), plt.title('Dilatación')
    plt.axis('off')

    plt.subplot(4, 4, 13), plt.imshow(apertura, cmap='gray'), plt.title('Apertura')
    plt.axis('off')

    plt.subplot(4, 4, 14), plt.imshow(cierre, cmap='gray'), plt.title('Clausura')
    plt.axis('off')

    plt.subplot(4, 4, 15), plt.imshow(gradient_image, cmap='gray'), plt.title('Gradiente Morfologico')
    plt.axis('off')

    plt.subplot(4, 4, 16), plt.imshow(topHat, cmap='gray'), plt.title('Top Hat')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Aplicar los kernels
    



# Uso de la función

dir='C:/2025_01ENE/SanAntonio-20250103T112337Z-001/SanAntonio'

seleccionar_y_ecualizar_imagen_cv2(dir)


