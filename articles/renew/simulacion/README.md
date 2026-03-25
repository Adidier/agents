# Simulación — MADDPG para Microrredes Energéticas

Implementación completa de los experimentos del artículo:

> **"Multi-Agent Deep Deterministic Policy Gradient for Mixed Cooperative-Competitive MAS Microgrid Scenario for RES Integration and Energy Arbitrage"**

Fuente: `articles/renew/energyIntegrationMicrogridEnergyTrading_w_ReinforcementLearning.pdf`  
Traducción y explicación: `articles/renew/MADDPG_traduccion_explicacion.md`

---

## Estructura del directorio

```
simulacion/
│
├── __init__.py              # Documentación del paquete
├── sim_config.py            # Configuración global y bootstrap de sys.path
├── benchmarks.py            # Agentes de referencia: RBM y MADQN
│
├── case_study_1.py          # Caso de Estudio 1: control HESS (solo ESS)
├── case_study_2.py          # Caso de Estudio 2: HESS + MGA + 5 xMGs
│
├── compare_algorithms.py    # Ejecuta todos los algoritmos y genera Tabla 3
├── visualize.py             # Genera gráficas (Fig 5a, Fig 5b, barras)
├── run_simulation.py        # Punto de entrada principal (CLI)
│
└── results/                 # Resultados JSON/CSV/PNG generados automáticamente
```

**Código RL subyacente** (en `src/maddpg_microgrid/`):
- `environment.py` — Entorno de microrred HESS con ESS, PV, WT, MGA, xMGs
- `maddpg.py` — Sistema MADDPG (entrenamiento centralizado, ejecución descentralizada)
- `ddpg.py`, `d3pg.py`, `td3.py` — Implementaciones de agente único
- `networks.py` — Actor, Critic, NoisyNet, D3PG distribucional
- `config.py` — Parámetros del papel (Tabla 1, Tabla 2)

---

## Configuración del entorno

### Requisitos
```bash
pip install torch numpy matplotlib
```
*(desde `/home/solar/dev/agents/`)*

---

## Uso rápido

Todos los comandos se ejecutan desde este directorio (`simulacion/`):

```bash
cd /home/solar/dev/agents/articles/renew/simulacion
```

### Prueba rápida (5 episodios, ~/30 segundos)

```bash
python run_simulation.py --compare --case 1 --episodes 5
```

### Un solo algoritmo — Caso de Estudio 1

```bash
python run_simulation.py --case 1 --algo maddpg --episodes 200
```

### Full paper replication — Caso de Estudio 1 (Table 3)

```bash
python run_simulation.py --compare --case 1 --episodes 200
```

### Caso de Estudio 2 — HESS + MGA + 5 xMGs

```bash
python run_simulation.py --compare --case 2 --episodes 200
```

### Algoritmos específicos

```bash
python run_simulation.py --compare --case 1 --algos maddpg mad3pg matd3 --episodes 200
```

### Visualizar resultados guardados

```bash
python run_simulation.py --visualize results/comparison_cs1_ep200.json --case 1
```

---

## Descripción del sistema (del artículo)

### Microrred principal

```
    [Utility Grid] ──── transformer ─────── [AC Bus]
                                              │    │
                          [WT] ──transformer──┘    │
                                                   │
                         [PV] ──inverter── [DC Bus] ──inverter──[AC Bus]
                                              │
                    [HESS: LIB + VRB + SC] ───┘

    [AC Bus] ──── transformer ──── [5 xMGs]
```

### Sistema de Almacenamiento Híbrido (HESS)

| ESS | Uso          | η_RTE | η_SDC  | Ciclos | P_CPC |
|-----|-------------|-------|--------|--------|-------|
| LIB | Mediano plazo | 95%  | 99.99% | 5 000  | £40   |
| VRB | Largo plazo  | 80%  | 100%   | 10 000 | £40   |
| SC  | Corto plazo  | 95%  | 99%    | 100 000| £6    |

Capacidad máxima por ESS: **2 MWh** | Potencia máxima: **1 MW**

### Generación renovable

| Fuente | Capacidad máx. | Modelo                           |
|--------|----------------|----------------------------------|
| PV     | 5 MW           | Radiación solar → potencia        |
| WT     | 2 MW (2×1 MW)  | Curva de potencia viento (Eq. 30) |

### Precios de energía

- **Compra** (mercado mayorista dinámico): £16–£144/MWh  
- **Venta** (feed-in tariff fijo): £16/MWh

---

## Algoritmos implementados

### Controladores de un solo agente (SGC)

| Algoritmo | Descripción |
|-----------|-------------|
| **DDPG**  | Actor-crítico determinista, base del artículo |
| **D3PG**  | DDPG distribucional — estima distribución Z(s,a) en lugar de E[Q] |
| **TD3**   | DDPG con doble crítico — reduce sobreestimación de Q |

### Multi-agente MADDPG (aprendizaje centralizado, ejecución descentralizada)

| Algoritmo  | Base  | Descripción |
|-----------|-------|-------------|
| **MADDPG** | DDPG  | Cada agente tiene actor local + crítico centralizado |
| **MAD3PG** | D3PG  | MADDPG con crítico distribucional |
| **MATD3**  | TD3   | MADDPG con doble crítico por agente |

**Recompensa de contribución marginal** (Sección 3.4.3):  
Cada agente ESS recibe como recompensa los ahorros globales *si ese ESS hubiera permanecido inactivo*, midiendo así su contribución marginal al objetivo común.

### Benchmarks (solo Caso de Estudio 1)

| Algoritmo | Descripción |
|-----------|-------------|
| **MADQN** | DQN independiente por agente, espacio de acciones discretas (5 acciones) |
| **RBM**   | Reglas deterministas: carga cuando RES > demanda (prioridad SC→LIB→VRB) |

---

## Caso de Estudio 2 — Subasta de energía

```
MGA decide:  volumen a vender + precio de reserva
     │
     ▼
Subasta secuencial (Algorithm 2):
  1. Ordenar xMGs por precio de puja (mayor primero)
  2. Si precio_puja > precio_reserva → asignar energía
  3. Energía no vendida → devuelta a la red a £16/MWh

xMG decide:  volumen de compra + precio de puja
xMG reward = ahorro vs comprar todo a la red eléctrica (£144/MWh)
```

---

## Resultados esperados (de la Tabla 3 del artículo)

*(Con datos reales de la Universidad de Keele, 200 episodios — nuestros resultados usan datos sintéticos)*

| Algoritmo | Ahorros brutos (£k) | Ahorros ajustados (£k) | vs DDPG  |
|-----------|---------------------|------------------------|----------|
| DDPG      | 42.92               | 34.39                  | —        |
| D3PG      | 48.78               | 40.78                  | +18.6%   |
| TD3       | 46.69               | 39.93                  | +16.1%   |
| **MADDPG**    | **59.58**       | **53.24**              | **+54.8%** |
| MAD3PG    | 59.50               | 53.44                  | +55.4%   |
| MATD3     | 56.19               | 50.06                  | +45.6%   |
| MADQN     | 47.28               | 36.32                  | +5.6%    |
| RBM       | 40.93               | 33.02                  | −4.0%    |

**Clave**: el enfoque multi-agente con recompensa de contribución marginal (MADDPG/MAD3PG) supera en ~55% al controlador global único (DDPG).

---

## Parámetros de entrenamiento (Tabla 2)

| Hiperparámetro              | Valor  |
|----------------------------|--------|
| Actor learning rate        | 5×10⁻⁴ |
| Critic learning rate       | 1×10⁻³ |
| Dimensiones capas ocultas  | 256    |
| Activación capas ocultas   | ReLU   |
| Batch size                 | 64     |
| Tau (actualización suave)  | 0.005  |
| Factor de descuento γ      | 0.99   |
| D3PG límites V_min, V_max  | −10, 10|
| TD3 delay del actor        | 2      |
| Inicio de aprendizaje      | paso 500|
| Acciones aleatorias hasta  | paso 1000|
| Total episodios            | 200    |
| Evaluación desde episodio  | 100    |

---

## Referencia

Si utilizas este código, cita el artículo original:

> "Multi-Agent Deep Deterministic Policy Gradient for Mixed Cooperative-Competitive MAS  
> Microgrid Scenario for RES Integration and Energy Arbitrage",  
> Applied Energy, 2023.
