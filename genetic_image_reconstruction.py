#!/usr/bin/env python
# coding: utf-8

"""
Genetic Algorithm for Image Reconstruction

Evolves a population of random pixel arrays toward a target image using
evolutionary operators: tournament/elitist selection, single-point and
uniform crossover, random and swap mutation, and elitism-based replacement.

Color complexity is reduced by simplifying pixel values to binary (0 or 255)
before evolution, which shrinks the search space dramatically while preserving
recognisable structure.
"""

from __future__ import annotations

import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image(file: str, size: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """Load an image file, resize it, and return as a NumPy array.

    Args:
        file: Path to the image file.
        size: Target (width, height) to resize the image to.

    Returns:
        A tuple of (pixel_array, is_png) where *pixel_array* is a NumPy array
        of the resized image and *is_png* indicates whether the file is PNG.
    """
    image = Image.open(file)
    image = image.resize(size)
    pixels = np.array(image)
    is_png = file.lower().endswith(".png")
    return pixels, is_png


def simplify_colors(image: np.ndarray) -> np.ndarray:
    """Flatten every colour channel to binary: 0 or 255.

    Values <= 127 become 0; values > 127 become 255.  This reduces the
    colour palette to at most 2^C colours (C = number of channels).

    Args:
        image: NumPy array of pixel data (modified in-place).

    Returns:
        The same array with simplified colour values.
    """
    image[image <= 127] = 0
    image[image > 127] = 255
    return image


# ---------------------------------------------------------------------------
# Search-space analysis
# ---------------------------------------------------------------------------

def compute_search_space_size(file: str) -> tuple[list[int], list[int]]:
    """Compute the search-space size for an image at increasing resolutions.

    For each resolution from 2 to 600 (step 16), the search space is
    estimated as *num_pixels * num_unique_colours*.  Two lists are returned:
    one for the original image and one for the colour-simplified version.

    Args:
        file: Path to the image file.

    Returns:
        A tuple (combinations, simple_combinations) of integer lists.
    """
    combinations: list[int] = []
    simple_combinations: list[int] = []

    for size in range(2, 600, 16):
        img, _ = load_image(file, (size, size))
        img_simplified = simplify_colors(np.copy(img))

        num_pixels = img.shape[0] * img.shape[1]

        if img.ndim == 3:
            num_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            num_simple_colors = len(
                np.unique(img_simplified.reshape(-1, img_simplified.shape[2]), axis=0)
            )
        else:
            num_colors = len(np.unique(img))
            num_simple_colors = len(np.unique(img_simplified))

        combinations.append(num_pixels * num_colors)
        simple_combinations.append(num_pixels * num_simple_colors)

    return combinations, simple_combinations


def plot_search_space_size(
    combinations_png: list[int],
    combinations_jpg: list[int],
    simple_combinations_png: list[int],
    simple_combinations_jpg: list[int],
) -> None:
    """Plot how the search-space size varies with image resolution.

    Compares original vs colour-simplified images for both PNG and JPEG
    inputs.

    Args:
        combinations_png:          Search-space sizes for the original PNG.
        combinations_jpg:          Search-space sizes for the original JPEG.
        simple_combinations_png:   Search-space sizes for the simplified PNG.
        simple_combinations_jpg:   Search-space sizes for the simplified JPEG.
    """
    x_values = range(2, 600, 16)
    plt.plot(x_values, combinations_png, label="Combinations PNG")
    plt.plot(x_values, combinations_jpg, label="Combinations JPG")
    plt.plot(x_values, simple_combinations_png, label="Simple Combinations PNG")
    plt.plot(x_values, simple_combinations_jpg, label="Simple Combinations JPG")
    plt.xlabel("Size")
    plt.ylabel("Search space")
    plt.title("Search space vs size")
    plt.ylim(bottom=0, top=3.0e9)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Genotype helpers
# ---------------------------------------------------------------------------

def compute_genotypes(
    image: np.ndarray, png: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique colour genotypes and their probabilities from an image.

    Args:
        image: Pixel array (should already have simplified colours).
        png:   Whether the source image is PNG (unused, kept for API compat).

    Returns:
        A tuple (genotypes, probabilities) where *genotypes* is an array of
        unique colour vectors and *probabilities* holds the relative
        frequency of each genotype.
    """
    flat_image = image.reshape(-1, image.shape[-1])
    genotypes, counts = np.unique(flat_image, axis=0, return_counts=True)
    probabilities = counts / flat_image.shape[0]
    return genotypes, probabilities


def generate_random_image(
    image: np.ndarray, genotypes: np.ndarray, prob: np.ndarray
) -> np.ndarray:
    """Create a random image using genotype probabilities from a target image.

    Each pixel is sampled independently from the genotype distribution.

    Args:
        image:     Target image (used only for its shape).
        genotypes: Array of unique colour vectors.
        prob:      Probability for each genotype.

    Returns:
        A randomly generated image with the same dimensions as *image*.
    """
    total_pixels = image.shape[0] * image.shape[1]
    random_indices = np.random.choice(len(genotypes), size=total_pixels, p=prob)
    random_image = genotypes[random_indices].reshape(image.shape).astype(np.uint8)
    return random_image


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def fitness_function(member: np.ndarray, goal: np.ndarray, png: bool = True) -> int:
    """Compute the fitness (distance) between a candidate and the goal image.

    Fitness equals the total number of differing colour-channel values after
    both images are colour-simplified.  Lower is better.

    Args:
        member: Candidate image array.
        goal:   Target image array.
        png:    Whether the source is PNG (unused, kept for API compat).

    Returns:
        Integer count of mismatched channel values.
    """
    simple_member = simplify_colors(np.copy(member))
    simple_goal = simplify_colors(np.copy(goal))
    return int(np.sum(simple_member != simple_goal))


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def initial_population(
    goal: np.ndarray, n_population: int, png: bool = True
) -> np.ndarray:
    """Generate an initial population of random images.

    Each individual is created by sampling pixels from the genotype
    distribution of the goal image.

    Args:
        goal:         Target image used to derive genotype probabilities.
        n_population: Number of individuals in the population.
        png:          Whether the source is PNG (unused, kept for API compat).

    Returns:
        A NumPy array of shape (n_population, H, W, C).
    """
    genotypes, prob = compute_genotypes(goal)
    population = np.array(
        [generate_random_image(goal, genotypes, prob) for _ in range(n_population)]
    )
    return population


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def selection_tournament(
    population: np.ndarray, scores: list[int], k: int = 10
) -> list[np.ndarray]:
    """Select parents via tournament selection.

    For each slot in the new population, *k* individuals are sampled at
    random and the one with the lowest (best) fitness wins.

    Args:
        population: Current population array.
        scores:     Fitness score for each individual.
        k:          Tournament size.

    Returns:
        A list of selected parent arrays.
    """
    selected: list[np.ndarray] = []
    for _ in range(len(population)):
        indices = random.sample(range(len(population)), k)
        best_idx = min(indices, key=lambda i: scores[i])
        selected.append(population[best_idx])
    return selected


def selection_truncation(
    population: np.ndarray, scores: list[int], k: int = 10
) -> np.ndarray:
    """Select the top-*k* individuals by fitness (truncation selection).

    Args:
        population: Current population array.
        scores:     Fitness score for each individual.
        k:          Number of parents to keep.

    Returns:
        An array containing the *k* fittest individuals.
    """
    sorted_indices = np.argsort(scores)
    return population[sorted_indices[:k]]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossover_single_point(
    p1: np.ndarray, p2: np.ndarray, r_cross: float
) -> tuple[np.ndarray, np.ndarray]:
    """Single-point column crossover between two parents.

    A random column index is chosen; the left part of one parent is combined
    with the right part of the other (and vice-versa).

    Args:
        p1:      First parent image array.
        p2:      Second parent image array.
        r_cross: Probability that crossover occurs.

    Returns:
        Two offspring arrays (c1, c2).
    """
    if np.random.rand() < r_cross:
        point = np.random.randint(1, p1.shape[1])
        c1 = np.concatenate((p1[:, :point], p2[:, point:]), axis=1)
        c2 = np.concatenate((p2[:, :point], p1[:, point:]), axis=1)
    else:
        c1, c2 = p1.copy(), p2.copy()
    return c1, c2


def crossover_uniform(
    p1: np.ndarray, p2: np.ndarray, r_cross: float
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each element is independently taken from either parent.

    A random boolean mask determines, element-by-element, which parent
    contributes to each offspring.

    Args:
        p1:      First parent image array.
        p2:      Second parent image array.
        r_cross: Probability that crossover occurs.

    Returns:
        Two offspring arrays (c1, c2).
    """
    if np.random.rand() < r_cross:
        mask = np.random.randint(2, size=p1.shape).astype(bool)
        c1 = np.where(mask, p1, p2)
        c2 = np.where(~mask, p1, p2)
    else:
        c1, c2 = p1.copy(), p2.copy()
    return c1, c2


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutation_random(
    descendant: np.ndarray,
    r_mut: float,
    genotypes: np.ndarray,
    prob: np.ndarray,
) -> np.ndarray:
    """Randomly mutate a fraction of pixels to new genotype values.

    Each mutated pixel is reassigned a colour sampled from the genotype
    probability distribution.

    Args:
        descendant: Image array to mutate (modified in-place).
        r_mut:      Fraction of pixels to mutate.
        genotypes:  Available colour genotypes.
        prob:       Probability for each genotype.

    Returns:
        The mutated image array.
    """
    total_pixels = descendant.shape[0] * descendant.shape[1]
    num_mutations = int(total_pixels * r_mut)
    indices = np.random.choice(total_pixels, num_mutations, replace=False)
    mutation_values = np.random.choice(len(genotypes), size=num_mutations, p=prob)
    flat = descendant.reshape(-1, descendant.shape[-1])
    flat[indices] = genotypes[mutation_values]
    return descendant


def mutation_swap(
    descendant: np.ndarray,
    r_mut: float,
    genotypes: np.ndarray,
    prob: np.ndarray,
) -> np.ndarray:
    """Swap-based mutation: randomly selected pixel pairs exchange values.

    Args:
        descendant: Image array to mutate (modified in-place).
        r_mut:      Fraction of pixels involved in swaps.
        genotypes:  Available colour genotypes (unused, kept for API compat).
        prob:       Probability for each genotype (unused, kept for API compat).

    Returns:
        The mutated image array.
    """
    total_pixels = descendant.shape[0] * descendant.shape[1]
    num_mutations = int(total_pixels * r_mut)
    indices = np.random.choice(total_pixels, num_mutations, replace=False)
    for i in range(0, len(indices) - 1, 2):
        descendant.flat[indices[i]], descendant.flat[indices[i + 1]] = (
            descendant.flat[indices[i + 1]],
            descendant.flat[indices[i]],
        )
    return descendant


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------

def replacement_full(
    population: np.ndarray,
    descendants: np.ndarray,
    r_replace: float,
) -> np.ndarray:
    """Replace the entire population with the new descendants.

    Args:
        population:  Current population (unused).
        descendants: New generation of individuals.
        r_replace:   Replacement rate (unused; full replacement is applied).

    Returns:
        A copy of *descendants*.
    """
    return descendants.copy()


def replacement_elitism(
    population: np.ndarray,
    descendants: np.ndarray,
    r_replace: float,
    goal: np.ndarray,
    elitism_rate: float = 0.002,
) -> np.ndarray:
    """Elitist replacement: preserve the best individuals from the old generation.

    The top *elitism_rate* fraction of the current population is carried
    over unchanged; the rest are filled from *descendants*.

    Args:
        population:    Current population array.
        descendants:   New offspring array.
        r_replace:     Replacement rate (unused; elitism handles balancing).
        goal:          Target image for fitness evaluation.
        elitism_rate:  Fraction of the population preserved as elites.

    Returns:
        New population array of the same size as *population*.
    """
    num_elites = int(elitism_rate * len(population))
    fitness_scores = [fitness_function(ind, goal) for ind in population]
    elite_indices = np.argsort(fitness_scores)[:num_elites]

    new_population = np.concatenate((population[elite_indices], descendants), axis=0)
    return new_population[: len(population)]


# ---------------------------------------------------------------------------
# Main genetic algorithm
# ---------------------------------------------------------------------------

def genetic_algorithm(
    goal: np.ndarray,
    n_iter: int,
    n_pop: int,
    r_cross: float,
    r_mut: float,
    r_replace: float,
    png: bool = True,
) -> tuple[Optional[np.ndarray], float]:
    """Run the genetic algorithm to reconstruct a target image.

    Pipeline per generation:
        1. Evaluate fitness of every individual.
        2. Select parents via truncation selection.
        3. Produce offspring through uniform crossover.
        4. Apply random mutation to each offspring.
        5. Replace most of the population (elitism preserves the best).

    Args:
        goal:      Target image to reconstruct.
        n_iter:    Number of generations to evolve.
        n_pop:     Population size.
        r_cross:   Crossover probability.
        r_mut:     Mutation rate (fraction of pixels mutated).
        r_replace: Replacement rate (passed to replacement operator).
        png:       Whether the source image is PNG.

    Returns:
        A tuple (best_image, best_fitness) of the best solution found.
    """
    genotypes, prob = compute_genotypes(goal, png)
    population = initial_population(goal, n_pop)

    best: Optional[np.ndarray] = None
    best_eval = float("inf")

    for generation in range(n_iter):
        scores = [fitness_function(ind, goal) for ind in population]

        # Track the overall best individual
        gen_best_idx = int(np.argmin(scores))
        if scores[gen_best_idx] < best_eval:
            best = population[gen_best_idx].copy()
            best_eval = scores[gen_best_idx]

        # Selection
        selected = selection_truncation(population, scores, k=10)

        # Crossover and mutation
        descendants: list[np.ndarray] = []
        while len(descendants) < len(population):
            p1 = selected[np.random.randint(len(selected))]
            p2 = selected[np.random.randint(len(selected))]
            c1, c2 = crossover_uniform(p1, p2, r_cross)
            c1 = mutation_random(c1, r_mut, genotypes, prob)
            c2 = mutation_random(c2, r_mut, genotypes, prob)
            descendants.extend([c1, c2])

        # Replacement (elitism)
        population = replacement_elitism(
            population, np.array(descendants[:n_pop]), r_replace, goal
        )

    return best, best_eval


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FILE_PNG = "logo_gray_scale.png"
    FILE_JPG = "logo_color.jpeg"

    # --- Search-space analysis ---
    combinations_png, simple_combinations_png = compute_search_space_size(FILE_PNG)
    combinations_jpg, simple_combinations_jpg = compute_search_space_size(FILE_JPG)
    plot_search_space_size(
        combinations_png, combinations_jpg,
        simple_combinations_png, simple_combinations_jpg,
    )

    # --- Reconstruct PNG image ---
    goal_png, is_png = load_image(FILE_PNG, size=(65, 25))
    best_png, score_png = genetic_algorithm(
        goal_png, n_iter=500, n_pop=650, r_cross=0.9, r_mut=0.005, r_replace=1.0
    )
    print(f"PNG best fitness: {score_png}")
    plt.imshow(best_png)
    plt.title("Best PNG reconstruction")
    plt.show()

    # --- Reconstruct JPEG image ---
    goal_jpg, is_jpg = load_image(FILE_JPG, size=(80, 45))
    best_jpg, score_jpg = genetic_algorithm(
        goal_jpg, n_iter=750, n_pop=650, r_cross=0.9, r_mut=0.0005,
        r_replace=1.0, png=False,
    )
    print(f"JPEG best fitness: {score_jpg}")
    plt.imshow(best_jpg)
    plt.title("Best JPEG reconstruction")
    plt.show()
