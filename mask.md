---
title: "Estudio de Mask R-CNN"
author: [Javier Sáez, Ismael Sánchez]
date: "23-01-2020"
titlepage: true
titlepage-color : 1e90ff
titlepage-text-color :  000000
titlepage-rule-color :  000000
---

# Introducción

La segmentación de instancias en imágenes es uno de los problemas más importantes en *Computer Vision*.

# Primeros pasos

**Definición.-** La *segmentación de instancias* en una imagen consiste en, para cada objeto dentro de una imagen, detectarlo y delimitar la zona de la imagen que ocupa.

![Instance Segmentation](images/iseg.png)

Existen diferentes modelos de *CNNs* que son capaces de obtener muy buenos resultados tanto en detección de objetos como en segmentación de imágenes, como *Fast/Faster R-CNN*. Es más complicado hacer la segmentación de instancias, pues requiere que nuestra red neuronal realice con precisión ambas tareas anteriores.

## Mask R-CNN

El modelo que estudiaremos es conocido como **Mask R-CNN**, y lo que pretende es extender *Faster R-CNN* añadiendo capas a este para predecir las máscaras de segmentación de cada una de las regiones de interés (*RoI*) de la imagen, a la vez que trata de dar *bounding boxes* y clasificación a los objetos de la imagen.

Sabemos que *Faster R-CNN* da, para cada candidato de objeto, dos salidas:

- Una etiqueta de clasificación
- Un *bounding* box

Con *Mask R-CNN* se le añade una nueva salida que es la máscara de segmentación del objeto.

Explicación rápida de  Faster R-CNN (dos líneas) y luego de cómo se amplía para mask R-CNN
## El modelo de CNN



## Subsection
