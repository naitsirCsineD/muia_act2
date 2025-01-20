import os
import random
import cv2
import numpy as np
from fpdf import FPDF
from skimage.metrics import structural_similarity as ssim



class Procesador1:

    def __init__(self, directory):
        if not os.path.exists(directory):
            raise ValueError(f"El directorio '{directory}' no existe.")
        self.directory = directory
        self.images = []
        self.operations_log = []

    def img_ref(self):
        self.clayminerals=None
        self.ferrousminerals=None
        self.hidrothermal=None
        self.ironoxides=None

    def seleccionar_y_ecualizar_imagen_cv2(self, img_file):
        """
        Carga una imagen desde un archivo y la prepara para procesamiento.
        """
        ruta_imagen = os.path.join(self.directory, img_file)

        try:
            print(f"Procesando imagen: {img_file}")
            img = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)  # Lee la imagen con su profundidad original

            if img is None:
                print("No se pudo cargar la imagen.")
                return None

            # Verificar si es de 16 bits
            if img.dtype == 'uint16':
                print("La imagen es de 16 bits. Normalizando y convirtiendo a 8 bits para procesar...")
                img = (img / 256).astype('uint8')  # Normalizar a 8 bits

            # Convertir a escala de grises si es necesario
            if len(img.shape) == 3:
                print("La imagen es en color. Convirtiendo a escala de grises...")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            return img
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
            return None

    def apply_gaussian_blur(self, img,kernel_size=(5,5)):
        params = {"ksize": kernel_size, "sigmaX": 0}
        self.operations_log.append({"operation": "Gaussian Blur", "params": params})
        return cv2.GaussianBlur(img, params["ksize"], params["sigmaX"])

    def apply_median_blur(self, img):
        params = {"ksize": 5}
        self.operations_log.append({"operation": "Median Blur", "params": params})
        return cv2.medianBlur(img, params["ksize"])

    def apply_edge_detection(self, img):
        params = {"threshold1": 100, "threshold2": 200}
        self.operations_log.append({"operation": "Edge Detection (Canny)", "params": params})
        return cv2.Canny(img, params["threshold1"], params["threshold2"])

    def apply_histogram_equalization(self, img):
        params = {}
        self.operations_log.append({"operation": "Histogram Equalization", "params": params})
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(img)

    def apply_sobel_edges(self, img):
        params = {"ksize": 3}
        self.operations_log.append({"operation": "Sobel Edges", "params": params})
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=params["ksize"])
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=params["ksize"])
        sobel = cv2.magnitude(sobel_x, sobel_y)
        return cv2.convertScaleAbs(sobel)
    
    def apply_realce_nitidez(self,img,kernel=np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])):
        params=kernel
        self.operations_log.append({"operation": "Realcede Nitidez", "params": params})
        kernel_realce_nitidez=cv2.filter2D(img,-1,params)
    
        gaussian = cv2.GaussianBlur(img, (9, 9), 5)
        unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        return unsharp


    def apply_ErosMorf(self,img,kernel_size=3):
        params=kernel_size
        self.operations_log.append({"operation": "Erosion Morfologica", "params": params})
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(img, kernel, iterations=1)


    def apply_Dilatation(self,img, kernel_size=3):
        '''Potencia y aumenta los contornos, haciendo que lod detalles se magnifiquen
        capaz de fusuonar objetos separados'''
        params=kernel_size
        self.operations_log.append({"operation": "Dilatacion", "params": params})
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        return cv2.dilate(img, kernel, iterations=1)


    def apply_apertura(self,img, kernel_size=3):
        '''Útil para separa objetos, elimina detalles en función del elemento estructural
        Se eliminan detalles, y se magnifica lo que queda'''
        params=kernel_size
        self.operations_log.append({"operation": "Apertura", "params": params})
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def apply_cierre(self, img,kernel_size=3):
        '''consolidar formas y rellenar pixeles en una imagen'''
        params=kernel_size
        self.operations_log.append({"operation": "Cierre", "params": params})
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def apply_GradMorf(self, img, kernel_size=3):
        """
        Calcular el gradiente morfológico (dilatación - erosión)
        """
        params = kernel_size
        self.operations_log.append({"operation": "Gradiente Morfologica", "params": params})
        erosion = self.apply_ErosMorf(img, kernel_size)
        dilation = self.apply_Dilatation(img, kernel_size)
        return cv2.subtract(dilation, erosion)


    def apply_HPF(self,img,kernel_size=(5,5)):
        params = {"ksize": kernel_size, "sigmaX": 0}
        self.operations_log.append({"operation": "HPF", "params": params})
        return img-cv2.GaussianBlur(img, params["ksize"], params["sigmaX"])


    def apply_logK(self,img,k=30):
        params = {"k":k}
        self.operations_log.append({"operation": "Tranformacion Logaritmica", "params": params})
        img = img.astype(np.float32)
        transformed_image=float(k) * np.log(1 + img)
        return cv2.normalize(transformed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def det_similarity(self,img1,img2):
        similarity_index, diff = ssim(img1, img2, full=True)
        return similarity_index, diff

    def process_sequence(self, img, max_operations=10):
        """
        Aplica una secuencia de transformaciones aleatorias a una imagen.
        """
        operations = [
            self.apply_gaussian_blur,
            self.apply_median_blur,
            self.apply_edge_detection,
            self.apply_histogram_equalization,
            self.apply_sobel_edges,
            self.apply_realce_nitidez,
            self.apply_ErosMorf,
            self.apply_Dilatation,
            self.apply_apertura,
            self.apply_cierre,
            self.apply_GradMorf,
            self.apply_HPF,
            self.apply_logK
        ]

        sequence_log = []
        processed_img = img.copy()
        num_operations = random.randint(1, max_operations)

        for _ in range(num_operations):
            operation = random.choice(operations)
            processed_img = operation(processed_img)
            sequence_log.append(self.operations_log[-1])

        return processed_img, sequence_log

    def combine_rgb_images(self, img_r, img_g, img_b):
        """
        Combina tres imágenes en un solo canal RGB.
        """
        # Asegurar que las imágenes tengan el mismo tamaño
        height, width = img_r.shape[:2]
        img_g = cv2.resize(img_g, (width, height)) if img_g.shape[:2] != (height, width) else img_g
        img_b = cv2.resize(img_b, (width, height)) if img_b.shape[:2] != (height, width) else img_b
        return cv2.merge((img_b, img_g, img_r))



    def save_plafs_to_pdf(self, pdf_path, num_plafs=100, max_operations=10):
        """
        Genera `num_plafs` secuencias únicas y las guarda en un PDF.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        archivos = [archivo for archivo in os.listdir(self.directory) if archivo.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]

        claysim=0
        ferroussim=0
        hidrotsim=0
        IronOxsim=0

        for plaf_idx in range(num_plafs):
            selected_files = random.sample(archivos, 3)

            # Seleccionar tres imágenes únicas
            original_imgs = [self.seleccionar_y_ecualizar_imagen_cv2(file) for file in selected_files]

            if not all(img is not None for img in original_imgs):
                print(f"No se pudo procesar el PLAF {plaf_idx + 1} debido a imágenes faltantes.")
                continue



            orginales_combinadas =self.combine_rgb_images(*original_imgs)


            # Procesar cada imagen con transformaciones únicas
            processed_imgs = []
            
            logs = []
            for img in original_imgs:
                processed_img, log = self.process_sequence(img, max_operations=max_operations)
                
                
                #claysim,difer=self.det_similarity(processed_img,self.clayminerals)
                #ferroussim,difer=self.det_similarity(processed_img,self.ferrousminerals)
                #hidrotsim,difer=self.det_similarity(processed_img,self.hidrothermal)
                #IronOxsim,difer=self.det_similarity(processed_img,self.ironoxides)


                processed_imgs.append(processed_img)
                
                
                logs.append(log)

            # Combinar imágenes procesadas en RGB
            combined_img = self.combine_rgb_images(*processed_imgs)
            

            # Agregar una página al PDF
            pdf.add_page()
            pdf.set_font("Arial", size=8)

            # Tabla 4x3
            table_x_start = 10
            table_y_start = 20
            cell_width = 60
            cell_height = 30

            # Primera fila: Imágenes originales
            for i, img in enumerate(original_imgs):
                img_path = f"original_temp_{plaf_idx}_{i}.jpg"
                cv2.imwrite(img_path, img)
                pdf.image(img_path, x=table_x_start + i * cell_width, y=table_y_start, w=cell_width, h=cell_height)

            # Segunda fila: Nombres de las imágenes originales
            for i, file_name in enumerate(selected_files):
                pdf.set_xy(table_x_start + i * cell_width, table_y_start + cell_height)
                pdf.multi_cell(cell_width, 5, txt=file_name, align='C')

            # Tercera fila: Imágenes procesadas
            for i, img in enumerate(processed_imgs):
                img_path = f"processed_temp_{plaf_idx}_{i}.jpg"
                cv2.imwrite(img_path, img)
                pdf.image(img_path, x=table_x_start + i * cell_width, y=table_y_start + 2 * cell_height, w=cell_width, h=cell_height)

            # Cuarta fila: Secuencias realizadas
            for i, log in enumerate(logs):
                pdf.set_xy(table_x_start + i * cell_width, table_y_start + 3 * cell_height)
                sequence_text = "\n".join([f"{op['operation']}" for op in log])
                pdf.multi_cell(cell_width, 5, txt=sequence_text, align='C')

            # Imagen combinada debajo de la tabla
            combined_path = f"combined_temp_{plaf_idx}.jpg"
            cv2.imwrite(combined_path, combined_img)
            pdf.image(combined_path, x=10, y=table_y_start + 4 * cell_height + 10, w=150, h=60)

            # Imagen combinada debajo de la tabla
            combined_orig__path = f"combined_orig_temp_{plaf_idx}.jpg"
            cv2.imwrite(combined_orig__path, orginales_combinadas)
            pdf.image(combined_orig__path, x=10, y=table_y_start + 5 * cell_height + 40, w=150, h=60)


        # Guardar el PDF
        pdf.output(pdf_path)

# Uso del sistema
directory = "C:\\MUIA2024\\MUIA, VA\\MUIA_VA_ACT02\\SanAntonio"
processor = Procesador1(directory)
processor.save_plafs_to_pdf("output_rgb_100_plafs.pdf", num_plafs=100, max_operations=10)
print("Reporte guardado en output_rgb_100_plafs.pdf")
