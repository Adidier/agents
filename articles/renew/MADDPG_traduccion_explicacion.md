# Traducción y Explicación: MADDPG en Microrredes Energéticas

---

## Contribuciones principales del artículo

- **Presenta un uso novedoso de MADDPG** en un escenario de microrred de Sistemas Multi-Agente (MAS) mixto, tanto cooperativo como competitivo, para la integración de Fuentes de Energía Renovable (RES) y el arbitraje energético. El aprendizaje centralizado y la ejecución descentralizada del método permite que cada agente evalúe su política observando las acciones de todos los demás agentes, sin necesidad de compartir parámetros ni redes críticas. Esto es ideal tanto para el HESS trabajando de forma cooperativa como para agentes con intereses propios orientados a hacer lo mismo con microrredes externas.

- **Considera tanto el arbitraje energético como la integración de RES** en un único problema dentro del MAS. Un esquema de precios en tiempo real del mercado energético mayorista asimétrico implica que los agentes deben responder tanto a los precios fluctuantes como a la producción de RES para reducir de forma óptima las facturas energéticas de la microrred.

- **Evalúa el uso de un controlador global de agente único versus múltiples agentes distribuidos** para el control de diferentes componentes de la microrred. La función de recompensa individual que cada agente recibe de las redes críticas individuales bajo MADDPG utiliza el concepto de contribución marginal para evaluar mejor cómo las acciones de los agentes impactaron el objetivo conjunto de reducir los costes energéticos. Esto también permite que cada agente en el HESS desarrolle su propia política única para adaptarse mejor a ese tipo de sistema de almacenamiento de energía (ESS).

---

## 2.1. Descripción de la red energética

La microrred principal está equipada con un HESS (Sistema Híbrido de Almacenamiento de Energía), así como generación fotovoltaica (PV) y eólica (WT). Está conectada a la red eléctrica principal, que establece precios de energía dinámicos desde los cuales la microrred puede importar energía o venderla de vuelta a una tarifa de inyección fija. Hay líneas de corriente alterna (AC) y continua (DC) presentes, conectadas mediante inversores, así como transformadores entre la microrred principal y la red eléctrica y la turbina eólica. Además, la microrred principal también está conectada a cinco microrredes externas (xMGs) a través de otro transformador; esas xMGs están conectadas a la red eléctrica principal pero no al mercado energético mayorista.

---

### 2.1.1. Sistema de Almacenamiento de Energía (ESS)

La microrred principal consta de un HESS con tres tipos diferentes de ESS:

| Tipo | Nombre completo | Uso óptimo | Características |
|------|----------------|------------|----------------|
| **LIB** | Batería de Iones de Litio | Almacenamiento a mediano plazo | La más barata por capacidad, baja autodescarga, buena eficiencia de ida y vuelta, pero pocos ciclos de vida totales |
| **VRB** | Batería de Flujo de Vanadio Redox | Almacenamiento a largo plazo | Autodescarga despreciable, pero baja eficiencia de ida y vuelta |
| **SC** | Supercondensador | Almacenamiento a corto plazo (fluctuaciones diarias) | Muchísimos ciclos de vida (muy barato de operar), pero alta autodescarga |

Los tres tienen la misma capacidad máxima y potencia máxima, y pueden transferir energía entre sí sin necesidad de pasar por un inversor.

La política de control que el agente debe aprender debe gestionar de forma óptima las características contrastantes de los tipos de ESS para maximizar sus fortalezas y mitigar sus debilidades respectivas.

---

### 2.1.2. Demanda y Fuentes de Energía Renovable

Los datos de demanda recopilados en el Campus de la Universidad de Keele abarcan desde las 00:00 del 1 de enero de 2014 hasta las 23:00 del 31 de diciembre de 2017. Los datos brutos están separados en diferentes emplazamientos residenciales, industriales y comerciales, así como lecturas de edificios clave. La demanda utilizada para esta simulación proviene de las tres subestaciones de entrada principales al campus, con las lecturas semiehorales sumadas para coincidir con la frecuencia de los datos meteorológicos horarios.

Esta microrred considera tanto la generación fotovoltaica (PV) como la eólica (WT), con la producción simulada utilizando datos meteorológicos recopilados en la estación meteorológica de la Universidad de Keele.

Los componentes de la red están conectados mediante inversores entre líneas AC y DC, así como transformadores para elevar o reducir voltajes.

---

### 2.1.3. Precios del mercado energético mayorista

La microrred puede comprar energía del mercado mayorista bajo un esquema de **precios dinámicos en tiempo real**, pero solo puede venderla de vuelta a la red eléctrica a una **tarifa de inyección fija constante**.

Al reaccionar al esquema de precios dinámico asimétrico (donde comprar es más caro que vender), aprender tanto a comprar energía cuando el precio mayorista es bajo como a maximizar la utilización de las RES será clave para el rendimiento de los agentes. Recompensar a los agentes en función de sus ahorros energéticos fomenta tanto el arbitraje energético efectivo como la integración de RES.

---

### 2.1.4. Comercio con microrredes externas

En un segundo caso de estudio, el agente agregador de microrred (**MGA**) vende energía a cinco xMGs más pequeñas. El MGA decide la cantidad de energía a vender en la próxima hora a las xMGs, que luego entran en una **fase de subasta** para competir por la energía.

Las xMGs pujan por una cantidad de energía al precio que están dispuestas a pagar. El MGA luego vende la energía a las xMGs que pujen más alto hasta que no quede nada por vender. El MGA también puede establecer un **precio de reserva** que debe alcanzarse para vender la energía; de lo contrario, se devuelve a la red eléctrica.

---

## 2.2. Metodología de control

El **Control Predictivo de Modelos (MPC)** ha sido históricamente el método más utilizado para un sistema de control, y es muy popular en la industria energética. Los métodos basados en modelos lograrán a menudo grandes resultados si se proporciona un modelo y datos precisos, pero los resultados solo son tan precisos como el modelo mismo, y los datos necesarios para crearlo no siempre están disponibles.

En contraste, el **Aprendizaje por Refuerzo (RL)** es un área de investigación en rápida expansión y la mayoría de los métodos son **sin modelo**, por lo que no requieren ningún modelo del entorno ni información previa sobre él, siendo mucho más flexibles. Por ello, este artículo se centra en el uso de RL.

Un desafío fundamental para este escenario específico es la **eficiencia de muestreo**, ya que solo hay un conjunto de datos, por lo que los agentes solo recibirán una pasada por el entorno. Por tanto, se requieren algoritmos RL conocidos por su eficiencia de muestreo para lograr ahorros energéticos superiores.

---

## 3. Metodología de Aprendizaje por Refuerzo

### 3.1. Fundamentos

En cada paso de tiempo $t$ en RL, el agente observa el estado actual del entorno $s_t$ y selecciona una acción $a_t$ del espacio de acciones siguiendo una política aprendida. Luego recibe una recompensa $r_t$ y transiciona a un nuevo estado $s_{t+1}$, con el objetivo de aprender una política que maximice su recompensa futura total con descuento.

Los algoritmos RL se pueden categorizar en:

| Tipo | Descripción |
|------|-------------|
| **Métodos de función de valor** | Asignan valor a cada estado estimando $V(s)$ o el par estado-acción $Q(s,a)$ |
| **Métodos de gradiente de política** | Parametrizan directamente la política mediante una red neuronal, ajustando los parámetros $\theta$ para maximizar $J(\theta)$ siguiendo el gradiente $\nabla_\theta J(\theta)$ |
| **Métodos Actor-Crítico** | Combinan ambos enfoques: aprenden una función de valor y parametrizan la política |

---

### 3.2. Algoritmos

#### 3.2.1. DDPG — *Deep Deterministic Policy Gradient*

DDPG es un método **actor-crítico** basado en los principios de Q-learning, donde:

- La **red actora** $\mu_\phi(s)$ (con pesos $\phi$) selecciona las acciones
- La **red crítica** $Q_\theta(s,a)$ (con pesos $\theta$) evalúa el rendimiento del agente

**Selección de acciones:** La política es determinista, por lo que el agente explora añadiendo un proceso de ruido $w$ a la salida del actor:

$$a_t = \mu_\phi(s_t) + w$$

Este ruido $w$ puede ser Gaussiano (ruido no correlacionado) o el proceso de Ornstein-Uhlenbeck (ruido correlacionado). En este trabajo se utilizan **capas NoisyNet** — el ruido se introduce en la estimación de la función de valor y el agente aprende a aumentarlo o disminuirlo.

**Memoria de repetición de experiencias (*Experience Replay Buffer*):** Tras ejecutar una acción, el agente almacena la tupla de transición $\langle s_t, a_t, r_t, s_{t+1}, d_t \rangle$ en un buffer, que se muestrea durante el entrenamiento. Esto convierte al algoritmo en un método **off-policy** (se entrena con muestras de políticas anteriores), lo que lo hace más eficiente en muestras que métodos on-policy como A3C o PPO. Esto es crucial aquí, ya que los agentes solo realizan un barrido de los 4 años de datos.

**Redes objetivo (*Target Networks*):** Se usan redes actora objetivo $\hat{\mu}_{\hat\phi}$ y crítica objetivo $\hat{Q}_{\hat\theta}$ con pesos fijos durante el entrenamiento para estabilizar el aprendizaje (evitar oscilaciones o divergencia).

**Actualización del crítico:** Los valores objetivo $y$ se estiman a partir de las estimaciones de la red crítica objetivo:

$$y_i = r_i + \gamma \hat{Q}_{\hat\theta}(s_{i+1}, \hat{\mu}_{\hat\phi}(s_{i+1}))$$

La red crítica se actualiza por descenso de gradiente minimizando el error cuadrático medio:

$$\mathcal{L}_i(\theta) = \mathbb{E}\left[(y_i - Q_\theta(s_i, a_i))^2\right]$$

**Actualización del actor** mediante ascenso de gradiente del gradiente de política determinista:

$$\nabla_\phi J_i(\phi) = \mathbb{E}\left[\nabla_a Q_\theta(s_i, a)\big|_{a=\mu(s)} \nabla_\phi \mu_\phi(s_i)\right]$$

**Actualizaciones suaves** de las redes objetivo al final de cada paso con parámetro de suavizado $\tau$:

$$\hat\theta \leftarrow \tau\theta + (1-\tau)\hat\theta$$
$$\hat\phi \leftarrow \tau\phi + (1-\tau)\hat\phi$$

---

#### 3.2.2. D3PG — *Distributional DDPG*

Un problema del DDPG estándar es que el **valor esperado** de $Q(s,a)$ puede **sobre-generalizar** en entornos con funciones de recompensa no deterministas (con aleatoriedad, como precios de energía variables).

**D3PG** elimina el paso de expectativa de la ecuación de Bellman y en su lugar estima la **distribución de valores completa** $Z(s,a)$:

> En lugar de predecir *"el valor promedio esperado"*, predice *"toda la distribución de posibles retornos"* — esto es más rico en información y más robusto ante la incertidumbre del entorno.

---

## Resumen General

> El artículo aplica **MADDPG** para gestionar una microrred híbrida con almacenamiento de energía (litio + vanadio + supercondensador), generación solar y eólica, y comercio con microrredes externas. Los agentes aprenden de forma autónoma a **ahorrar dinero comprando energía barata, maximizando el uso de renovables y vendiendo energía a microrredes vecinas mediante subastas**, sin necesitar un modelo matemático del entorno.

| Concepto | Significado clave |
|----------|------------------|
| **MADDPG** | Múltiples agentes RL que aprenden juntos pero actúan por separado |
| **HESS** | Sistema de almacenamiento híbrido (3 tecnologías complementarias) |
| **Arbitraje energético** | Comprar barato, almacenar, usar o vender cuando conviene |
| **RES** | Fuentes renovables (solar + eólica) cuya producción es impredecible |
| **xMG** | Microrredes externas que compiten por comprar energía sobrante |
| **Experience Replay** | Mecanismo que permite re-aprender de experiencias pasadas |
| **Actor-Crítico** | Arquitectura RL: el actor decide, el crítico evalúa |
