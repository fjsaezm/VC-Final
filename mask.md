---
title: "Estudio de Mask R-CNN"
author: [Javier Sáez, Ismael Sánchez]
date: "23-01-2020"
titlepage: true
titlepage-color : 1e90ff
titlepage-text-color :  000000
titlepage-rule-color :  000000
---

\tableofcontents
\newpage

# Introducción {.unnumbered}

La segmentación de instancias en imágenes es uno de los problemas más importantes en *Computer Vision*. En el siguiente documento, trataremos de detallar el intento que hemos realizado de tomar un esqueleto de una **Mask R-CNN** , adaptar la base de datos de entrada y ejecutar el entrenamiento sobre este esqueleto.

Además, la idea principal consiste en que una vez que se haya conseguido entrenar sobre el modelo base, realizar diversas modificaciones tanto sobre el conjunto de datos como sobre el propio esqueleto de la red  para tratar de mejorar los resultados obtenidos sobre nuestro conjunto concreto de datos.

Los resultados no fueron los esperados, debido a las complicaciones que obtuvimos en creación de nuestra propia base de datos para el problema. Estas complicaciones serán detalladas durante todo el documento.

# Primeros pasos

**Definición.-** La *segmentación de instancias* en una imagen consiste en, para cada objeto dentro de una imagen, detectarlo y delimitar la zona de la imagen que ocupa.

![Instance Segmentation](images/iseg.png){ width=50% }

Existen diferentes modelos de *CNNs* que son capaces de obtener muy buenos resultados tanto en detección de objetos como en segmentación de imágenes, como *Fast/Faster R-CNN*. Es más complicado hacer la segmentación de instancias, pues requiere que nuestra red neuronal realice con precisión ambas tareas anteriores.

## Mask R-CNN

El modelo que estudiaremos es conocido como **Mask R-CNN**, y lo que pretende es extender *Faster R-CNN* añadiendo capas a este para predecir las máscaras de segmentación de cada una de las regiones de interés (*RoI*) de la imagen, a la vez que trata de dar *bounding boxes* y clasificación a los objetos de la imagen.

Sabemos que *Faster R-CNN* da, para cada candidato de objeto, dos salidas:

- Una etiqueta de clasificación
- Un *bounding* box

Con *Mask R-CNN* se le añade una nueva salida que es la máscara de segmentación del objeto.

![Mask R-CNN framework](images/framework.png){ width=50% }


**Faster R-CNN** tiene dos etapas: una primera en la que se dan propuestas de *bounding boxes* para los objetos, y la segunda (que es en sencia *Fast-RCNN*), extrae características usándo **RoIPooling** por cada propuesta que ha obtenido y luego realiza clasificación y regresión sobre las *bounding-box*.

**Mask R-CNN** tiene las dos mismas etapas, con la diferencia que en la segunda, en *paralelo* a predecir la clase y la *bounding-box*, también aporta una máscara de segmentación para cada *RoI*.

Duranete el entrenamiento, se define una **función de pérdida** con múltiples factores por cada *RoI*, tenemos la función:
$$
L = L_{cls} + L_{box} + L_{mask}
$$

En *Faster-RCNN* teníamos sumando la función de pérdida de la clasificación ($L_{cls}$) y la de la *bounding-box* ($L_{box}$). Ahora, añadimos un sumando a la función, la función de pérdida de la máscara de segmentación de cada objeto ($L_{mask}$).

Es importante denotar que gracias a $L_{mask}$, las máscaras y las predicciones de clases están desacopladas, al contrario que cuando se aplican *FCN* para segmentación semántica. Este desacople ayuda mucho a una buena segmentación de instancias.

Existen además multitud de redes que se han entrenado , dando como resultado vectores de pesos que se utilizan para, dado una imagen nueva, hacer una predicción sobre ella y sus objetos. Algunos ejemplos conocidos son *ImageNet* o *COCO*. En nuestro caso, utilizaremos los pesos que nos proporciona **MS-COCO**.


# El dataset: Open Images Dataset V5

Lo primero que tuvimos que pensar tras escoger el tema del trabajo, fue qué dataset escoger para el entrenamiento de nuestra red neuronal. A propuesta del profesor, decidimos utilizar [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html). En concreto, comenzamos a explorar la versión número *5* de este conjunto de datos, pues es en ella donde se ha introducido al dataset la información necesaria para la **segmentación de instancias**. Posee además información sobre *bounding-boxes*. Con la elección de este *dataset*, comenzó nuestra odisea de intentar obtener un subconjunto de todos los datos que contiene.

En un principio, este *dataset* contiene:

- Sobre 9 millones de imágenes
- 15,851,536 cajas en 600 categorías
- 2,785,498 instance segmentations en 350 categorías

La enorme cantidad de imágenes hace que solamente el conjunto de las imágenes pese $512GB$. Esto se nos antojaba desde el principio demasiada información, así que pensamos que podríamos tratar de obtener una partición de este subconjunto. 

Lo primero que hicimos fue ponernos en contacto con soporte de la página. Le comentamos que queríamos hacer un proyecto y que nos gustaría obtener un subconjunto del dataset porque el tamaño nos parecía demasiado elevado. La respuesta que obtuvimos andaba en las líneas de: "hoy día un disco duro de ese tamaño no es muy caro, comprad uno para guardar el dataset".

Ante esta respuesta, decidimos buscar por internet si alguien había hecho una herramienta para obtener una cantidad algo más pequeña de imágenes seleccionando, por ejemplo, ciertas clases. Primeramente, encontramos la herramienta [oiv4segmentation], que nos permitía obtener imágenes de la versión **4** del dataset que queríamos. Esta herramienta no nos permitía obtener la información que queríamos, pues no descargaba las máscaras de segmentación que nos hacían falta para nuestra *Mask R-CNN*. 

En los *issues* del repositorio, encontramos un enlace a [dirtySegmentationdownloaderOIV5], un script que nos permitía descargar las imágenes de la versión **5** de la base de datos, y nos daba algunas indicaciones para obtener información sobre las máscaras y los nombres de las clases en las que está organizada esta base de datos. Por cada clase hay muchas imágenes en el conjunto de datos, así que hicimos ciertas modificaciones sobre el script para descargar solo un número determinado de fotos (que establecimos a $1000$ por clase), y añadimos las clases que queríamos.

En la zona de descargas de la base de datos, encontramos el archivo *train-annotations-object-segmentation.csv*. Este contenía la información sobre cada una de las **máscaras de segmentación**, dividida en:

- *MaskPath*, que nos indica el nombre de la máscara 
- *ImageID*, el nombre de la imagen con la que va asociada
- *LabelName*, la clase a la que pertenece el objeto del cual es la máscara the MID of the object class this mask belongs to.
- *BoxID*, el identificador de la bounding box an identifier for the box within the image.
- *BoxXMin, BoxXMax, BoxYMin, BoxYMax*, coordenadas de la caja ligada a la máscara, normalizadas. Esta caja es desde donde se partió para hacer la máscara, no la *bounding-box*
- *Clicks*, que indican los clicks anotados por el usuario de forma manual para generar la máscara.

En un principio,pensamos que con estos clicks sería suficiente para generar las máscaras de cada imagen sin tener que descargar todos los archivos, así que cuando programamos la clase, hicimos una zona de código en la que a partir de las coordenadas de los clicks, obteníamos la máscara. Sin embargo, debido a que en ocasiones los clicks no se tomaban en orden,las máscaras tenían formas extrañas y no representaban bien al objeto que querían señalar.


Una vez teníamos este archivo, pudimos descargar las imágenes con este código:

```python
f=pd.read_csv("train-annotations-object-segmentation.csv")
name_num = [["Person",['/m/01g317']],["TrafficLight",['/m/015qff']],["Car",['/m/0k4j']],["TrafficSign",['/m/01mqdt']]]
threads = 20
nImages = 1000
commands = []
for el in name_num:
    u = f.loc[f['LabelName'].isin(el[1])]
    pool = ThreadPool(threads)
    for ind in u.index[0:nImages]:
        image = u['ImageID'][ind]

        download_dir = "/home/fjsaezm/oiv5segmentation/images/"

        # Train images
        path = "train" + '/' + str(image) + '.jpg ' + '"' + download_dir + '"'
        command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/' + path
        if not command in commands:
            commands.append(command)

list(tqdm(pool.imap(os.system, commands), total = len(commands)))
```

Como se puede ver, tomabamos las mil primeras imágenes que podíamos encontrar de cada clase, generábamos un comando para descargarla mediante *amazon web service*, y la descargábamos individualmente. En el propio script en el que nos basamos, se nos indicaba que teníamos que descargar las imágenes de las máscaras de forma individual. Estas estaban divididas en 15 archivos *.zip*, de unos *400MB* cada uno. Cada archivo *zip* comienza con un número del *0* al *9* o una letra de la *a* a la *f*, y contiene en él todas las máscaras de las imágenes que empiezan por la misma letra que ese archivo.

Cuando descargamos las imágenes, nos dimos cuenta de que las imágenes no venían clasificadas por temas. Es decir, todas las imágenes que tienen por ejemplo una *Manzana*, no empiezan por el mismo número. Esto supuso un problema, pues nos obligaba a descargar todos los archivos con las máscaras, descomprimirlos, subirlos a *Drive* para poder usarlos en *Google Collaboratory*, en el cual sabemos que el espacio del que disponemos es limitado. Además, sabemos que los tiempos de entrenamiento de las redes neuronales cuando tienen muchas capas y/o muchas imágenes con las que entrenar son muy grandes, por lo que no era eficiente para el proceso de experimentación tener un número muy elevado de imágenes, además de que *Google Collaboratory* nos dio innumerables **timeouts** al intentar trabajar con directorios con un numero tan grande de ficheros.

Es por ello que decidimos hacer por nuestra cuenta un segundo script, que buscase en las imágenes aquellas que empezasen por $0,1$ ó $2$, y las copiamos en una nueva carpeta, haciendo así una partición más pequeña del conjunto inicial, pero que pensamos que nos sería más fácil de manejar.
```python
onlyfiles = [f for f in listdir("images/") if isfile(join("images/", f))]
for a in onlyfiles:
    if a[0] == '0' or a[0] == '1' or a[0] == '2':
        path = "images/"+a
        shutil.copy(path,target_dir)

```
Este tenía una segunda parte en la que , para no subir a *Drive* todas las máscaras de todas las imágenes que empezasen por $0,1$ ó $2$, seleccionaba las máscaras de estas carpetas que empezaban por estos números y las copiaba en una carpeta nueva, para subir sólo la información verdaderamente relevante a *Drive*.
```python
onlyfiles = [f for f in onlyfiles if f[0] == '0' or f[0] == '1' or f[0] == '2']
masks0 = [f for f in listdir("../Downloads/train-masks-0/") if isfile(join("../Downloads/train-masks-0/",f))]
masks1 = [f for f in listdir("../Downloads/train-masks-1/") if isfile(join("../Downloads/train-masks-1/",f))]
masks2 = [f for f in listdir("../Downloads/train-masks-1/") if isfile(join("../Downloads/train-masks-1/",f))]

for img in onlyfiles:
    id = img.split(".")[0]
    if id[0] == '0':
        for m in masks0:
            if id in m:
                path = "../Downloads/train-masks-0/"+m
                shutil.copy(path,target_dir)
    if id[0] == '1':
        for m in masks1:
            if id in m:
                path = "../Downloads/train-masks-1/"+m
                shutil.copy(path,target_dir)
    if id[0] == '2':
        for m in masks2:
            if id in m:
                path = "../Downloads/train-masks-2/"+m
                shutil.copy(path,target_dir)
```


## El subconjunto de datos obtenido

Las clases que descargamos fueron:

- Persona
- Semáforo
- Señal de tráfico
- Coche

Las imágenes que obteníamos eran todas del conjunto de entrenamiento del *dataset*. Tratamos de seguir unas indicaciones que venían en el *script* de descarga de imágenes original para descargar imágenes para el conjunto de *validación* y el conjunto de *test*, pero al realizar el comando con *AWS*, nos daba un error en la descarga pues no existían los archivos.

Es por ello que decidimos tomar solo las imágenes del conjunto de entrenamiento y realizar nosotros mismos particiones aleatorias de este subconjunto de imágenes para obtener nuestros conjuntos de *train*, *validation* y *test*, aunque este último nunca lo llegamos a tomar si quiera pues no pudimos entrenar la red finalmente. Utilizamos el siguiente código que genera conjuntos de índices aleatorios para obtener nuestras particiones de *train* y *validation*:
```python
# Pre-known
num_imgs = 793
n_all_imgs = 100
n_train_imgs = 90
n_val_imgs = 10


# 1. Select 600 images
all_index = random.sample(range(num_imgs), n_all_imgs)
# 2. Split in 500-100
train_index = random.sample(all_index,n_train_imgs)
val_index = [i for i in all_index + train_index if i not in all_index or i not in train_index]

trainID = [new_csv.iloc[ind]['ImageID'] for ind in train_index]
valID = [new_csv.iloc[ind]['ImageID'] for ind in val_index]
```
Cuando habíamos tomado los índices, nos quedamos con los *ID* de las imágenes en ambos conjuntos. Además, nos aseguramos que ambos conjuntos fuesen disjuntos para no hacer la validación con elementos que habíamos utilizado en el entrenamiento.

# Mask R-CNN

## Esqueleto

Buscando por internet se pueden encontrar diversos modelos de *Mask R-CNN* que llevan debajo diferentes arquitecturas. En nuestro caso, decidimos usar una arquitectura que encontramos en *github*, desarrollada por [matterport]. Junto con el **esqueleto** de la *CNN*, se nos daban varios **ejemplos** de cómo utilizarlo creando un objeto que representase nuestro *Dataset*, que luego utilizaría el propio esqueleto en el entrenamiento. Esta clase contiene a todas nuestras imágenes y con ellas una serie de atributos bastante útiles en el entrenamiento de la red.

Este modelo de *Mask R-CNN* es diferente del primer modelo de este tipo de red que se [FAIR] propuso. En concreto, las principales diferencias las podemos encontrar en:

1. **Reescalado de imágenes**, que se realiza en esta red para poder hacer entrenamiento con múltiples imágenes en cada *batch*. Se preserva el *aspect ratio*.
2. **Bounding Boxes**, que son ignoradas (ya veremos como para generar nuestro *Dataset* no las utilizamos), sino que son generadas sobre la marcha. Se escoge la caja más pequeña que encapsula todos los píxeles de una máscara. Esto facilita la implementación y hace más facil aplicar aumentos en la imagen, que romperían las *bounding box* iniciales
3. **Learning rate**, que utilizan en este caso ratios más pequeños pues descubren que en algunos casos los pesos de la red podían explotar.

El modelo que utilizamos tiene un total de **224** capas, que están divididas en zonas. De hecho, en la implementación del modelo, se realizan diferentes clases y funciones para realizar cada una de las zonas de las capas de la red. No existe una documentación de cómo son las capas del modelo, hay que entrar en [model] y examinar el código para ver cómo está estructurado, tarea que resulta laboriosa debido a la gran extensión del archivo (casi $3K$ líneas).

Un dato relevante sobre el modelo es que puede usar dos **columnas vertebrales** (*backbone*) diferentes a la hora de entrenar:

- *resnet50*, que ya usamos en las prácticas y sabemos que tiene 50 capas.
- *resnet101*, que, como sabemos por su nombre, tendrá 101 capas.

Para obtener el modelo dentro de **Google Collaboratory** , podemos utilizar los comandos de *git* para clonarlo dentro del propio código y luego importar los archivos que podemos utilizar. Además del propio código del esqueleto de la red neuronal, al importarlo se obtienen más funcionalidades como un método ```visualize``` que nos da una utilidad para probar si los datos que estamos obteniendo son correctos. 

El código que hemos utilizado para importar las clases e instalar las dependencias es el siguiente:
```python
!rm -rf ~/Mask_RCNN
!git clone --quiet https://github.com/matterport/Mask_RCNN.git
%cd ~/Mask_RCNN

!pip install -q PyDrive
!pip install -r requirements.txt
!python setup.py install


from mrcnn.config import Config
from mrcnn import model as modellib, utils
```
    
# TrafficDataset

El modelo de red neuronal convolucional que utilizamos, se utiliza dándole como parámetros un objeto del tipo **utils.Dataset**. Este tiene por defecto implementadas muchas funciones que nos dan información sobre cada una de las imágenes que añadamos al conjunto de datos creado en *python*, así como una serie de atributos que se pueden consultar para trabajar con estas imágenes.

Es por ello que para adaptar el modelo, lo que debemos hacer es crear una clase 
```python
TrafficDataset(utils.Dataset)
```
que nos permita obtener un conjunto de entrenamiento, uno de validación y uno de test para entrenar el modelo.

Para ello, es necesario crear dos clases nuevas con relaciones de herencia sobre otras de las que heredan sus métodos, que sobreescribiremos para que se adapten a nuestro conjunto de datos. Estas son:

- ```TrafficConfig(Config)```

- ```TrafficDatabase(utils.Database)```

## TrafficConfig

La primera de las dos clases proporciona al modelo cierta configuración para el entrenamiento. Lo que hacemos en ella es definir una serie de parámetros que se utilizan a la hora de entrenar el modelo. Tomamos primero una de uno de los ejemplos y la adaptamos para que obedezca a los parámetros de nuestra base de datos.

Es importante saber que esta clase hereda de ```Config```, que la tenemos definida en el archivo ```config.py``` que obtenemos en el repositorio que hemos clonado.

```python
class TrafficConfig(Config):
    """Configuration for training on the birds dataset.
    Derives from the base Config class and overrides values 
    """
    # Give the configuration a recognizable name
    NAME = "traffic"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + Traffic Lights + People + Cars + Traffic Signs


    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small size, not a lot of data
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
    # Backbone. Can be resnet50 or resnet101
    BACKBONE = "resnet101"
```

Existen algunos parámetros más que podemos utilizar, pero no hemos podido probar el funcionamiento de los mismos debido a que no hemos podido entrenar la red al no consguir cargar bien el conjunto de datos. A destacar, tenemos:

- *LEARNING_RATE*, que hemos comentado antes que en el modelo afirman que es mejor que sea $LEARNING\_RATE < 0.02$. Podemos ajustarlo y ver cómo afecta esto al entrenamiento.
- *IMAGE_RESIZE_MODE*, que nos permite que las imágenes se transformen en el modelo. También se le puede poner una dimensión mínima y una máxima con *IMAGE_MIN_DIM/IMAGE_MAX_DIM*

Una vez que hemos definido nuestra configuración, tenemos que crear un objeto con este tipo para pasárselo al modelo y que conozca la configuración que queremos.
```python
config = TrafficConfig()
```
Podemos mostrar el resultado de la creación del objeto utilizando ```config.display()```. El resultado en nuestro caso, nos ofrece los siguientes datos (tomamos los más relevantes, pues la lista es demasiado larga):
```python
Configurations:
BACKBONE                       resnet101
BACKBONE_STRIDES               [4, 8, 16, 32, 64]
BATCH_SIZE                     8
DETECTION_MAX_INSTANCES        100
DETECTION_MIN_CONFIDENCE       0.7
DETECTION_NMS_THRESHOLD        0.3
FPN_CLASSIF_FC_LAYERS_SIZE     1024
GPU_COUNT                      1
GRADIENT_CLIP_NORM             5.0
IMAGES_PER_GPU                 8
IMAGE_CHANNEL_COUNT            3
IMAGE_RESIZE_MODE              square
LEARNING_MOMENTUM              0.9
LEARNING_RATE                  0.001
LOSS_WEIGHTS                   {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0}
MASK_POOL_SIZE                 14
MASK_SHAPE                     [28, 28]
NAME                           traffic
NUM_CLASSES                    5
ROI_POSITIVE_RATIO             0.33
TRAIN_ROIS_PER_IMAGE           200
VALIDATION_STEPS               50
WEIGHT_DECAY                   0.0001
```

## TrafficDataset

Por último, tenemos que definir la clase que contendrá la información de las imágenes y que generará un *dataset* con el que podremos realizar el entrenamiento. Para ello, debemos definir una clase 
```python
class TrafficDataset(utils.Dataset):
```
que, como vemos, hereda de *utils.Dataset*, del que obtendrá la mayoría de sus métodos. Para poder utilizarla, deberemos hacer *override* de tres métodos principales:

1. ```load_dataset(self,dataset_dir,subset_f)```
2. ```load_mask(self,image_id)```
3. ```load_image(self,image_id)```

Por último, hicimos también un pequeño *override* de la función ```image_reference(self,image_id)```, pero que no comentaremos mucho pues no es de gran relevancia.

Lo primero que comentaremos es que definimos una función que asigna a cada clase de los objetos que queríamos detectar un *id* numérico, pues el modelo trabaja con estos para las clases. El código es sencillo:
```python
def class_to_ind(c):
      if c == "/m/01g317": # Person
        return 1
      elif c == "/m/015qff": # Traffic Light
        return 2
      elif c == "/m/0k4j": # Car
        return 3
      elif c == "/m/01mqdt": # Traffic Sign
        return 4
      else:
        return -1
```

### load_dataset 

Una vez definida esta función, declaramos la clase y hacemos *override* de la primera función mencionada anteriormente:
```python
def load_dataset(self, dataset_dir, subset,f):
        "Add the images to the dataset"
        name_num = [["Person",['/m/01g317']],["TrafficLight",['/m/015qff']],["Car",['/m/0k4j']],["TrafficSign",['/m/01mqdt']]]
        # Add classes
        for i,clase in enumerate(name_num):
          self.add_class("traffic", class_to_ind(clase[1][0]), clase[0])

        # Add images 
        # Cojo todas las filas que tengan una clase de las cuales estudiamos
        u = f.loc[f['LabelName'].isin([clase[1][0] for clase in name_num])]
        # Para cada ID
        for image_id in subset:
          datos_imagen = u.loc[u['ImageID'].isin([image_id])]
          image_dir = "{}/{}.jpg".format(dataset_dir, image_id)
          image = skimage.io.imread(image_dir)
          height, width = image.shape[:2]
          self.add_image(
                "traffic",
                image_id= image_id,  # use file name as a unique image id
                path=image_dir,
                width=width, height=height )
```

Este método se llama justo después de crear el objeto de tipo *TrafficDataset* (también se podría llamar en ```__init__```, para evitar que llamarlo de forma explícita). Carga las imágenes y las clases en el dataset.

El procedimiento que se realiza es sencillo. Lo primero que se hace es añadir las clases usando un método de la clase padre. A continuación, toma una por una las imágenes , y toma la siguiente información de la misma para usarla como parámetros:

- Ruta
- *ID*
- Anchura y altura

Y añade la imagen llamando al método padre ```add_image``` con los parámetros que ha obtenido para cada imagen. 

Se pueden añadir más parámetros , como posibles ```polygons``` (o clicks que tengamos sobre la imagen) o ```clases``` que tengamos en la imagen.


### load_masks

Ahora, definimos otra función que nos sirve para que cuando se pida desde el modelo, se carguen las máscaras de segmentación de una imagen. El código que hemos implementado es el siguiente:

```python
def load_mask(self, image_id):

        info = self.image_info[image_id]
        mask = []
        class_ids = []
        maskfiles = [f for f in listdir(mask_dir) if isfile(join(mask_dir, f))]
        for m in maskfiles:
          mn = m.split(".")[0]
          if info["id"] in mn:
            maskclass = m.split("_")[1]
            id  = "/"+maskclass[0]+"/"+maskclass[1:len(maskclass)]
            print(id)
            if class_to_ind(id) > 0:
              class_ids.append(id)
              mask.append(m)
        
        count = len(mask)
        finalmask = []
        for index,item in enumerate(mask):
          path = mask_dir+"/"+item
          r = skimage.io.imread(path)
          finalmask.append(r)

        finalmask = np.stack(finalmask, axis=-1)

        #Occlusions
        occlusion = np.logical_not(finalmask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            finalmask[:, :, i] = finalmask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(finalmask[:, :, i]))

        return finalmask, class_ids
```

En este caso, la función tiene un cuerpo más largo pero el contenido es igual de sencillo. El procedimiento que se realiza es:

1. Cargar la información de la imagen, y las máscaras
2. Por cada máscara, comprobar si pertenece a la imagen, y si lo es, añadirla a un vector y la clase del objeto que sea la máscara, a otro vector
3. Por cada máscara seleccionada en el paso anterior, se lee el fichero de la misma y se añade a un vector de máscaras
4. Se tratan las oclusiones que puedan darse entre las máscaras

#### Los problemas con esta función

En una primera implementación que realizamos, tomábamos del archivo *csv* los clicks que estaban alrededor de cada objeto y generábamos las máscaras coloreando los píxeles que quedaban dentro del área delimitada por el polígono que resulta de unir los clicks.

Sin embargo, esta implementación no generaba bien las máscaras de las imágenes, pues posiblemente tomábamos mal el orden de los píxeles y las máscaras generadas eran figuras:

- Con bordes muy pronunciados
- Que no correspondían nada bien con el objeto que se quería segmentar

Podemos observar un ejemplo de esto en la siguiente imagen:

![Error en máscara de segmentación](images/error.jpg)

Fue por ello que tuvimos que realizar todo el cambio en la estructura, y pasar de no subir las máscaras a *Drive* a subirlas y cambiar por tanto esta función para que leyese la máscara necesaria en cada caso.

Esta nueva implementación nos ayudó a obtener unos mejores resultados en obtención ed las máscaras.

**Observación.-** Haciendo pruebas para ver qué estaba sucediendo, hemos llegado a la conclusión de que posiblemente en ocasiones los puntos se tomasen desordenados. De hecho, si tomamos unos puntos concretos para la máscara, digamos por ejemplo:
```
r=[30,190,30,190]
c=[30,190,190,30]
```
Estos son los puntos de un cuadrado desordenados. Esto es, si un cuadrado es *ABDC*, en este caso tendríamos los vértices en orden *ACBD*. Con esto, el resultado que obtenemos es el siguiente:
![Error en máscara de segmentación manual](images/cuadrado.jpg)

Donde podemos ver que los vértices se han unido con el siguiente en la lista, que no es con el que deberían unirse en una máscara real, por lo que de ahí podría venir el error.

### load_image

Esta función no tiene mucha relevancia, pero tuvimos que hacerle *override* pues lo solicitaba para hacer pruebas una vez que se habia creado el objeto de tipo *TrafficDataset* para poder probar que todo se está ejecutando correctamente. El código es muy sencillo:
```python
def load_image(self, image_id):
        """Load images according to the given image ID."""
        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'])
        return image
```

Más adelante veremos cómo utilizamos esta función

# El entrenamiento

Una vez que tenemos definida esta clase, podemos crear una instancia de la misma que contenga el conjunto de *train* y otra que contenga el conjunto de *validation* para poder entrenar el modelo. 

Lo primero que tenemos que hacer es crear un modelo tomado del paquete que hemos obtenido. Hay que crearlo en modo entrenamiento. Realizamos lo siguiente:
```python
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)
# Cargamos los pesos de Coco ya preentrenado
model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
```
En la segunda línea le cargamos los pesos de **MS-COCO**, como hemos indicado antes que haríamos. Además, hay que excluirle cierto tipo de capas como las que hemos excluido. Esto se debe a que serán diferentes debido a que las clases de objetos que queremos detectar no serán las mismas.

Tras haber cargado los pesos, podemos crear los dos datasets para *train* y validación* de la siguiente manera:
```python
# Set train
dataset_train = TrafficDataset()

informacion_imagenes = pd.read_csv(ANNO_DIR)
dataset_train.load_dataset(IMAGE_DIR, trainID, new_csv)
dataset_train.prepare()

# Validation dataset
dataset_val = TrafficDataset()
dataset_val.load_dataset(IMAGE_DIR, valID, new_csv)
dataset_val.prepare()
```
Esto nos prepara los datasets para que se los podamos pasar directamente como parámetros al modelo en la función que entrenará.

Al ejecutar estas líneas de código, no obtenemos ningún fallo aparente y, como veremos, podemos mostrar las imágenes y las máscaras que tienen, pero a la hora de entrenar obtendremos un fallo.

## Visualizando los datos

El archivo ```visualize.py```, como hemos comentado antes, nos proporciona utilidades para ver que todo esté ocurriendo con normalidad en cuanto a los datos se refiere. Existe un sencillo ejemplo de uso del mismo, que hemos utilizado para comprobar que las imágenes se está cargando bien

**Nota.-** Haciendo uso de esta función fue cuando nos dimos cuenta de que las máscaras mediante clicks no eran las correctas, y tuvimos que cambiar la implementación.

```python
# Inspect the train dataset
print("Image Count: {}".format(len(dataset_val.image_ids)))
print("Class Count: {}".format(dataset_val.num_classes))
for i, info in enumerate(dataset_val.class_info):
    print("{:3}. {:50}".format(i, info['name']))
  
# Load and display random samples
image_ids = np.random.choice(dataset_val.image_ids, 1)
for image_id in image_ids:
    image = dataset_val.load_image(image_id)
    mask, class_ids = dataset_val.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids,
                                dataset_val.class_names)
```
Utilizamos este código para:

- En la primera parte, comprobar que las clases, y el número de imágenes es el correcto.
- En la segunda parte, obtenemos un subconjunto aleatorio de tamaño que queramos, y mostramos tanto la imagen original como cada una de las máscaras con las clases.

La salida que obtenemos de este código es la siguiente:
```
Image Count: 1
Class Count: 5
  0. BG                                                
  1. Person                                            
  2. TrafficLight                                      
  3. Car                                               
  4. TrafficSign   
```
![Test validation](images/test.png)

Comentando primero que, para hacer la prueba, se había tomado como tamaño de conjunto de validación $n = 1$, vemos que se toma bien cuántas imágenes hay y las diferentes clases.

Además, se puede ver como carga bien la única máscara que tiene la imagen, que es la cara de una persona y su brazo.

## El entrenamiento

Una vez tenemos todos los elementos preparados, podemos pasar a entrenar la red. Nosotros , sin embargo, hemos obtenido un fallo en la creación del dataset que nos impide que la red pueda entrenar, pues da error con los ```class_id``` de las imágenes.

Para realizar el entrenamiento, como es ya sabido de prácticas anteriores, tenemos que realizar lo siguiente:

```python
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='all')
```
Con esto , el modelo entrenaría usando el dataset que hemos creado, tomando el ```LEARNING_RATE``` de nuestro configs, durante 2 épocas y todas las capas.

Sin embargo, debido a que hemos estado todo el tiempo trabajando para conseguir darle forma a los datos que se nos daban de forma tan desestructurada debido a su gran cantidad y formato, no lo hemos conseguido hasta el último momento, al intentarlo nos apareció un error que no hemos conseguido solucionar con el tiempo suficiente como para que pudiésemos entrenar la red.

La red comenzaba a entrenar y a los pocos segundos obteníamos el siguiente error:
```python
ERROR:root:Error processing image {'id': '23938d05c8b7588a', 'source': 'traffic', 'path': '/content/drive/My Drive/data/images2/23938d05c8b7588a.jpg', 'width': 1024, 'height': 678}
Traceback (most recent call last):
  File "/root/Mask_RCNN/mrcnn/model.py", line 1709, in data_generator
    use_mini_mask=config.USE_MINI_MASK)
  File "/root/Mask_RCNN/mrcnn/model.py", line 1265, in load_image_gt
    class_ids = class_ids[_idx]
TypeError: only integer scalar arrays can be converted to a scalar index
```

Que no conseguimos solucionar. Es posible que no sea un error de tener un fallo grande de concepto, pero la base de datos nos ha comido demasiado el tiempo.

## La evaluación de los datos

A pesar de no haber podido entrenar nuestra red, teníamos información de cuales eran los procedimientos que debíamos seguir para comprobar cómo de bueno estaba siendo el entrenamiento con nuestro conjunto de datos.

Lo primero que se debe hacer tras entrenar es siempre coger un conjunto de *test* y aplicarle un predictor, para ver qué información nos devuelve sobre nuestra imagen, es decir, qué objetos cree que hay en ella y sus máscaras de segmentación.

Para ello, debemos definir una nueva clase de configuración del modelo, en la que ahora el modo que tendremos no será de *training* como en el caso anterior, sino que será un modo de *inferencia* sobre la entrada.  La implementación es cortita:
```python
class InferenceConfig(TrafficConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
config = InferenceConfig()
```
Con esta nueva configuración, podemos tomar los pesos que hemos cargado y darle al modelo un modo de inferencia de la siguiente manera:
```python
model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
model.load_weights(model.find_last(),True)
```

De este modo, podremos aplicar la función predict sobre los datos de entrada para obtener la salida que ya hemos comentado. 

Ahora, una vez que tenemos el resultado de qué ha predicho nuestro modelo, ¿ cómo medimos la bondad del resultado ? Usamos la métrica **mAP**

### mAP

Esta métrica de precisión es muy utilizada en proyectos en los que se tratan de reconocer un conjunto de objetos. Esta viene definida como
$$
MAP = \frac{\sum_{q = 1}^Q AveP(q)}{Q}
$$
donde $Q$ es el número de elementos en el set, y $AveP(a)$ es la precisión media *AP* para un elemento del conjunto, $q$.

Para calcular esta métrica, disponemos de la función ```calculate_ap``` en el paquete ```utils.py```. Esto se calcula para una imagen. Cuando lo tenemos para todas las imágenes, aplicamos la fórmula anterior.

Esta función necesita un parámetro *IoU* (*intersection of union*), al que le podemos diferentes valores según nos interese obtener un resultado más o menos estricto.

Existe también una función que nos permite calcular todas las *mAP* en un rango, haciendo pequeños incrementos. Esta es ```calculate_ap_range```, disponible en el mismo paquete.


# Solucionando errores

## Errores en las máscaras

El tiempo dedicado a la base de datos nos desvió de la tarea central: **adaptar la Mask R-CNN a nuestra base de datos**. Los últimos avances que conseguimos nos permitían ya cargar las imágenes con sus respectivas máscaras proporcionadas.

Aquí nos surge otro problema:

![Not enough masks providen in image](images/nopeople.png)

Como podemos ver, hay muchas más personas en la imagen de las que se nos muestran visualizando la unión de las máscaras obtenidas. Esto es un gran error, pues al entrenar, cuando el modelo tome las máscaras de la imagen, no tendrá todas las personas y obtendrá por tanto una información errónea de cuándo existe o no una persona en nuestra foto.

Decidimos comprobar esto. Recordamos que los nombres de las imágenes de las máscaras empiezan por el nombre de la imagen a la que corresponden. Así, creamos este script para comprobar cuántas máscaras hay en una imagen determinada en la que hay personas. El código es simple, y el archivo lo llamamos ```test.py```:

```python
import os
from os.path import isfile,join
from os import listdir

i = 0

mask = "../../Downloads/train-masks-0/"
onlyfiles = [f for f in listdir(mask) if isfile(join(mask, f))]
id = "0a7e0b2c83069f3c"

for f in onlyfiles:
    if id in f:
        i = i+1

print(i)
```
Cuenta cuántas imágenes hay que contengan el id de la imagen. La imagen escogida es la siguiente:

![Lots of people](images/people.jpg){ width=50% }

Y, si ejecutamos el script, el resultado obtenido es:

```bash
[fjsaezm@fjsaezm VC-Final]$ python test.py 
8

```
Como vemos, tenemos muchas menos máscaras que personas. Esto es grave para el entrenamiento de nuestra red. Sin embargo, esto escapa de nuestras manos, pues esta información es la que se nos da desde la base de datos.

## Error en el entrenamiento

Haciendo referencia al error mencionado al intentar entrenar la red

```python
ERROR:root:Error processing image {'id': '23938d05c8b7588a', 'source': 'traffic', 'path': '/content/drive/My Drive/data/images2/23938d05c8b7588a.jpg', 'width': 1024, 'height': 678}
Traceback (most recent call last):
  File "/root/Mask_RCNN/mrcnn/model.py", line 1709, in data_generator
    use_mini_mask=config.USE_MINI_MASK)
  File "/root/Mask_RCNN/mrcnn/model.py", line 1265, in load_image_gt
    class_ids = class_ids[_idx]
TypeError: only integer scalar arrays can be converted to a scalar index
```

Este tenía fácil solución. Estábamos tomando como *id* de las imágenes su propio nombre. Sin embargo, estos eran alfanuméricos y la *API* que se nos proporciona solo nos permitía que fuesen numéricos. 
Así, tuvimos que hacer un pequeño cambio en la función ```load_dataset```, a la hora de recorrer las imágenes para introducirlas. Lo que hacemos es numerarlas según nos las encontremos, y asignar este número como **id**. El resultado es el siguiente:

```python
def load_dataset(self, dataset_dir, subset,f):
        "Add the images to the dataset"
        name_num = [["Person",['/m/01g317']],["TrafficLight",['/m/015qff']],["Car",['/m/0k4j']],["TrafficSign",['/m/01mqdt']]]
        # Add classes
        for i,clase in enumerate(name_num):
          self.add_class("traffic", class_to_ind(clase[1][0]), clase[0])

        # Add images 
        # Cojo todas las filas que tengan una clase de las cuales estudiamos
        u = f.loc[f['LabelName'].isin([clase[1][0] for clase in name_num])]
        # Cojo todos los ids de las imagenes y elimino los duplicados
        # Para cada ID
        for i,image_id in enumerate(subset):
          # Cojo las filas en las que aparece el ID
          datos_imagen = u.loc[u['ImageID'].isin([image_id])]
          image_dir = "{}/{}.jpg".format(dataset_dir, image_id)
          image = skimage.io.imread(image_dir)
          height, width = image.shape[:2]
          self.add_image(
                "traffic",
                image_id = i,  # use file name as a unique image id
                image_name = image_id,
                path=image_dir,
                clases = [class_to_ind(clase) for clase in datos_imagen['LabelName']],
                path_masks = [path for path in datos_imagen['MaskPath']])
```

# Resultados,propuestas de mejora y conclusión

## Resultados

Tras haber conseguido hacer pequeños entrenamientos de la red al solucionar los errores, intentamos hacer entrenamientos. Sin embargo, los resultados que obtenemos son pésimos. Esto se debe a que los entrenamientos requieren un número de imágenes mayor, además de que , como hemos comentado, no tenemos toda la información de las máscaras de una imagen. Es por ello que el modelo detectas cosas aleatorias en la imagen, que ni si quiera concuerdan con lo que debería ser realmente.

Algunos ejemplos de resultados son:

![Bad detection on people](images/peopledetection.jpg)

![Bad detection on a car](images/badcar.jpg)

Como podemos ver, los resultados dejan mucho que desear.

- Las máscaras de segmentación dadas son pésimas
- La certeza de que tenemos determinados objetos es bastante baja
- El modelo encuentra objetos que no existen

Esto es claramente debido al pequeño número de imágenes dadas, los errores en las máscaras y las pocas épocas que hemos podido dar al entrenamiento.


## Propuestas de mejora

Se proponen a continuación una serie de ideas que teníamos para tratar de mejorar los resultados que nos diese primeramente la red entrenada de la forma más básica.

1. La primera y la más obvia, es realizar el entrenamiento entrenando solo la última capa. Con esto dejamos que el modelo actúe de extractor de características y entrenamos la salida. Esto se haría sencillamente cambiando el parámetro en la función de entrenamiento de un modelo de la forma:
```python
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='heads')
```
El tiempo de entrenamiento sería muchísimo menor, ya que tendríamos que entrenar muchas menos capas. Sería un buen punto de partida para ver qué ocurre si no entrenamos nada de la red salvo la salida.

2. Profundizar un poco, congelando algunas capas más que sólamente las *heads*. El planteamiento es el siguiente, mantener las primeras capas congeladas, pues han entrenado mucho en bases de datos muy grandes y podemos presuponer que son una base sobre la que alzar nuestro castillo. Por ello, podemos congelar un número determinado de capas externas que son las que decidirán los detalles finales sobre los pesos de salida (esto es lo que se conoce como **fine tuning**). 
Por ejemplo, ya que nuestro modelo tiene *224* capas, podemos congelar un número significativo. Pongamos por ejemplo, un $10\%$ de la red, es decir, unas $22$ capas, las finales.
Esto podría hacer que nuestro modelo tras haber extraído las características iniciales , entrenase bien para ajustarse a los datos concretos de nuestro modelo. Se haría fácilmente de la siguiente manera.
```python
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='22+')
```

3. Comparar el resultado del congelamiento del $10%$ de las capas con el resultado de congelar tanto el $15\%$ como el $5\%$. Podríamos compararlos comparando la **mAP** de los tres casos. Esto nos podría ayudar a conocer si a la hora de ajustar el modelo, es necesario congelar algunas capas más, o hemos congelado capas de sobra.

4. Realizar *data augmention*. Nuestra cantidad de datos era bastante reducida por el número de archivos que *Drive* nos dejaba tener y por la lentitud de *Google Collaboratory* para procesarlos desde drive. Es por ello que utilizando aumento de datos como vimos en la práctica dos, podría hacer que el entrenamiento fuese más efectivo, teniendo más posibilidades en el entrenamiento del modelo. Esto se realizaría sencillamente de la siguiente forma:
```python
# Right/Left flip 50% of the time
augmentation = imgaug.augmenters.Fliplr(0.5)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='+22')
            #augmentation=augmentation)
```



## Conclusión

A pesar de la gran cantidad de tiempo dedicado al trabajo, la desorganización en los datos que teníamos que manejar nos ha retrasado más de lo esperado y no hemos podido obtener los resultados que queríamos.

Hemos experimentado cómo es ponerse al frente de una base de datos de tamaño considerable, como son $512GB$ sólo en el conjunto de entrenamiento, y también cómo aunque nuestros propios dispositivos tengan memoria suficiente como para guardar la base de datos entera, no serían capaces ni de terminar una época del entrenamiento de una red así antes de romperse por completo.

Sin embargo, hemos adquirido conocimientos sobre cómo enfrentarnos a estas bases de datos, sobre la estructura que tiene una *Mask R-CNN* y su funcionamiento y sobre como tendríamos que enfocar el problema de adaptar una base de datos nueva a un problema ya existente con unos pesos ya existentes, cosa que desconocíamos hasta el momento.


[oiv4segmentation]:https://github.com/EscVM/OIDv4_ToolKit
[dirtySegmentationdownloaderOIV5]:https://github.com/WyattAutomation/dirtySegmentationdownloaderOIV5/blob/master/dirtySegmentationJPGdl.py
[matterport]:https://github.com/matterport/Mask_RCNN
[FAIR]:https://arxiv.org/pdf/1703.06870.pdf
[model]:https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
