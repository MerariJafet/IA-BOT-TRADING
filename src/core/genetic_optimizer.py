"""
Genetic Optimizer - Motor de optimización evolutiva para estrategias.

Este módulo implementa un algoritmo genético para optimizar los parámetros
de las políticas de trading, evaluando cada generación mediante simulación
y evolucionando hacia mejores configuraciones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.logger import get_logger
from src.core.policy_agent import PolicyAgent
from src.core.simulation_environment import MarketSimulator

logger = get_logger(__name__)


class GeneticAlgorithmOptimizer:
    """Optimizador de algoritmo genético para parámetros de estrategia."""

    def __init__(
        self,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.2,
        sequence_path: str = "data/sequences/BTCUSDT_seq_1m.parquet",
    ):
        """
        Inicializa el optimizador genético.

        Args:
            population_size: Tamaño de la población
            generations: Número de generaciones a evolucionar
            mutation_rate: Probabilidad de mutación (0-1)
            sequence_path: Ruta a datos para simulación
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.sequence_path = sequence_path

        logger.info(
            f"🧬 GeneticAlgorithmOptimizer inicializado: "
            f"pop={population_size}, gen={generations}, mut={mutation_rate}"
        )

    def random_individual(self) -> Dict[str, float]:
        """
        Genera un individuo aleatorio (conjunto de parámetros).

        Returns:
            Diccionario con parámetros aleatorios
        """
        return {
            "threshold": np.random.uniform(0.4, 0.8),
            "lr": np.random.uniform(0.01, 0.1),
        }

    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """
        Aplica mutación a un individuo.

        Args:
            individual: Individuo a mutar

        Returns:
            Individuo mutado
        """
        ind = individual.copy()

        if np.random.rand() < self.mutation_rate:
            ind["threshold"] = np.clip(
                ind["threshold"] + np.random.uniform(-0.05, 0.05), 0.3, 0.9
            )

        if np.random.rand() < self.mutation_rate:
            ind["lr"] = np.clip(
                ind["lr"] + np.random.uniform(-0.01, 0.01), 0.001, 0.2
            )

        return ind

    def crossover(
        self, parent1: Dict[str, float], parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Realiza cruza entre dos padres.

        Args:
            parent1: Primer padre
            parent2: Segundo padre

        Returns:
            Hijo generado por cruza
        """
        child = {
            "threshold": (parent1["threshold"] + parent2["threshold"]) / 2,
            "lr": (parent1["lr"] + parent2["lr"]) / 2,
        }
        return self.mutate(child)

    def fitness(self, individual: Dict[str, float]) -> float:
        """
        Evalúa el fitness de un individuo.

        Args:
            individual: Individuo a evaluar

        Returns:
            Valor de fitness (PnL total)
        """
        try:
            # Crear agente con parámetros del individuo
            agent = PolicyAgent(threshold=individual["threshold"])

            # Simular estrategia
            sim = MarketSimulator(
                sequence_path=self.sequence_path,
                policy=agent.decide,
                capital=10000,
            )
            metrics = sim.run()

            # Usar PnL como fitness
            return metrics["pnl_total"]

        except Exception as e:
            logger.warning(f"Error evaluando fitness: {e}")
            return -np.inf

    def evolve(self) -> Tuple[Tuple[Dict[str, float], float], List[Dict[str, Any]]]:
        """
        Ejecuta el proceso evolutivo completo.

        Returns:
            Tupla con (mejor_individuo, fitness_mejor) y historial
        """
        logger.info("🚀 Iniciando evolución genética...")

        # Generar población inicial
        population = [self.random_individual() for _ in range(self.population_size)]

        best = None
        history = []

        for g in range(self.generations):
            logger.info(f"📊 Generación {g+1}/{self.generations}")

            # Evaluar fitness de toda la población
            scores = np.array([self.fitness(ind) for ind in population])

            # Ordenar por fitness (mayor a menor)
            ranked = sorted(
                zip(population, scores), key=lambda x: x[1], reverse=True
            )
            best = ranked[0]

            logger.info(
                f"✨ Gen {g+1}: Best PnL={best[1]:.4f}, "
                f"Threshold={best[0]['threshold']:.3f}, "
                f"LR={best[0]['lr']:.4f}"
            )

            # Selección: mantener la mitad superior
            num_survivors = max(2, self.population_size // 2)
            survivors = [x[0] for x in ranked[:num_survivors]]

            # Generar hijos mediante cruza
            children = []
            num_children = self.population_size - len(survivors)

            for _ in range(num_children):
                parent1 = survivors[np.random.randint(0, len(survivors))]
                parent2 = survivors[np.random.randint(0, len(survivors))]
                child = self.crossover(parent1, parent2)
                children.append(child)

            # Nueva población
            population = survivors + children

            # Guardar historial
            history.append(
                {
                    "generation": g + 1,
                    "best_pnl": float(best[1]),
                    "best_threshold": float(best[0]["threshold"]),
                    "best_lr": float(best[0]["lr"]),
                    "avg_pnl": float(np.mean(scores)),
                    "std_pnl": float(np.std(scores)),
                }
            )

        # Guardar reporte final
        report_path = Path("reports")
        report_path.mkdir(exist_ok=True)
        history_file = report_path / "genetic_optimization_history.json"

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(
            f"✅ Optimización genética completada. "
            f"Mejor threshold={best[0]['threshold']:.3f}, "
            f"LR={best[0]['lr']:.4f}, PnL={best[1]:.2f}"
        )
        logger.info(f"📄 Historial guardado en {history_file}")

        return best, history


if __name__ == "__main__":
    optimizer = GeneticAlgorithmOptimizer(
        population_size=8, generations=3, mutation_rate=0.2
    )
    best_individual, optimization_history = optimizer.evolve()

    print("\n🏆 Mejor configuración encontrada:")
    print(f"  Threshold: {best_individual[0]['threshold']:.3f}")
    print(f"  Learning Rate: {best_individual[0]['lr']:.4f}")
    print(f"  PnL: ${best_individual[1]:.2f}")
