# Obligatorio DeepLearning
## Universidad ORT Uruguay

El presente trabajo obligatorio tiene como objetivo la clasificación de secuencias de log de un HDFS en NORMALES y ANORMALES. 

Esta notebook presenta todo el código desarrollado para realizar el análisis y exploración de los datos; la construcción, entranemiento y evaluación de distintas arquitecturas de redes neuronales recurrentes y técnicas de búsqueda de parámetros y regularización. A su vez se presenta como documentación del trabajo resultante.

A lo largo de este trabajo presentaremos las siguientes arquitecturas:

* Baseline presentada por el cuerpo docente
* Modelo Mejorado basado en el baseline
* Modelo Mejorado en el cual se aplican hiperparámetros obtenidos mediante *grid search*
* Modelo Mejorado sobre el cual se aplica  *batch normalization* en base a los hiperparámetros obtenidos mediante *grid search*
* Modelo Mejorado entrenado con datos generados mediante *data augmentation* en base a los hiperparámetros obtenidos mediante *grid search*
* Modelo Mejorado regularizado mediante *gradient clipping* en base a los hiperparámetros obtenidos mediante *grid search*
* Modelo Mejorado regularizado mediante *gradient normalization* en base a los hiperparámetros obtenidos mediante *grid search*
* Modelo Mejorado regularizado mediante *gradient normalization* entrenado con datos generados mediante *data augmentation* en base a los hiperparámetros obtenidos mediante *grid search*
* Modelo Mejorado sobre el cual se aplica  *batch normalization* regularizado mediante *gradient normalization* en base a los hiperparámetros obtenidos mediante *grid search*

A lo largo del trabajo fuimos construyendo distintas arquitecturas que permitieron cumplir con las premisas planteadas en los requerimientos del obligatorio.
 
El trabajo no solo nos sirvió para profundizar en los distintos conceptos aprendidos en clase sino investigar técnicas adicionales como grid search, batch normalization, gradient clipping y gradient normalization.
 
Al momento de realizar el trabajo se optó por realizar ejecuciones con pocas epochs pero investigar todas las técnicas planteadas. ¿Por qué hicimos esto? Si bien la evidencia muestra que es posible lograr mejores resultados con una mayor cantidad de epochs entendimos que el trabajo presentaba la oportunidad de profundizar en las técnicas solicitadas. A esto se le suma los tiempos de entrenamiento que, para *grid search* rondaba el entorno de las 18 horas.
 
En cuanto a los modelos la siguiente tabla muestra los resultados finales obtenidos.
|index|Modelo|Accuracy|Precision|Recall|F1-score|
|---|---|---|---|---|---|
|5|Improved Model Grid Search with GN|0\.9989306926727295|0\.9851247673050674|0\.996943703452174|0\.990958303814308|
|2|Improved Model Grid Search|0\.9987126588821411|0\.981722643273391|0\.9968313102257391|0\.9891524015555664|
|3|Improved Model Grid Search with BN|0\.9986919164657593|0\.9814011180243527|0\.9968206061089357|0\.9889810600108891|
|8|Improved Model Grid Search with DA & GN|0\.9991382956504822|0\.9887225265823769|0\.9967167400158798|0\.9926850266636233|
|1|Improved Model|0\.9987126588821411|0\.9820219329541074|0\.9964973056214117|0\.9891452697636277|
|4|Improved Model Grid Search with GC|0\.9986607432365417|0\.9812173336061155|0\.9964705453294035|0\.9887168516592166|
|7|Improved Model Grid Search with DA|0\.9991175532341003|0\.9887042298547424|0\.9963720312947493|0\.9925062809357698|
|0|Initial Model|0\.9994289875030518|0\.9955231484983076|0\.9946955677229987|0\.9951089893153939|
|6|Improved Model Grid Search with BN & GN|0\.9978302121162415|0\.9956805261140256|0\.9669839800807936|0\.9808766509668125|

Puede apreciarse que para todos los modelos entrenados los resultados son satisfactorios. Si bien la diferencia entre ellos en términos generales está en las décimas, el modelo que obtuvo el mejor recall fue el mejorado usando gradient normalization.

Para este dataset y basándonos en el principio de la navaja de Occam nos quedamos con el modelo mejorado usando los hiperparámetros obtenidos mediante el grid search. La arquitectura de este modelo es relativamente simple y su costo de entrenamiento es bajo. A modo de experimento futuro se podría volver a entrenar este modelo usando una mayor cantidad de epochs.

En la misma línea el modelo baseline podría también ser entrenado con una gran cantidad de epochs (mayor a 2000). De todas formas entendemos que el objetivo pedagógico del trabajo no es tener modelos siendo entrenados durante horas sino comprender y aplicar los tópicos tratados en la materia y eso entendemos se ha logrado con éxito.
