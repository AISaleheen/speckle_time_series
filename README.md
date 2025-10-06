# speckle_time_series
# Analysis of Magnetic Speckle Dynamics

In this poject,  the dynamic behavior of magnetic domains using data from Coherent X-ray Scattering experiments is analyzed. The core of this technique relies on tracking the intensity fluctuations of magnetic speckles over time to characterize the underlying physical processes, such as the motion of domain walls and topological defects. The data consist of photon arrival times and coordinates, collected through a TimePix based photon counting detector. 

The analysis workflow is designed to transform raw photon event data  into quantitative metrics of the system's dynamics, namely its characteristic timescale and its "memory."

## Analysis Workflow

The analysis pipeline consists of the following key steps:

1.  **Speckle Identification:** Speckles are identified from the raw 2D scattering images as localized, high-intensity regions. These regions of interest are typically selected along a specific q-ring, corresponding to a characteristic length scale of the magnetic texture (e.g., the stripe domain width).

2.  **Time Series Extraction:** The photon counts within each identified speckle are integrated and binned over time to generate a time series. For this project, a bin size of 1 second is used, resulting in a series of intensity values `I(t)` for each speckle.

3.  **Autocorrelation Analysis:** The temporal dynamics are first characterized by calculating the Autocorrelation Function (ACF) for each time series. The ACF measures how quickly the magnetic configuration decorrelates, providing a characteristic timescale (`τ`) of the dynamics.

4.  **Fractal Analysis (Hurst Exponent):** To probe for long-range correlations and memory in the dynamics, the Hurst exponent (`H`) is calculated for each speckle's time series. This analysis reveals whether the dynamics are random, persistent, or anti-persistent, providing deep insight into the nature of the underlying physical motion.

## Key Concepts

### Magnetic Speckles
A magnetic speckle pattern is a complex interference pattern generated when coherent X-rays scatter from a material's magnetic domain structure. Each pattern serves as an interference fingerprint of a material's magnetic texture. Its temporal fluctuations provide a direct, real-time measurement of the underlying magnetic dynamics.


### Autocorrelation Function (ACF)
The ACF measures the self-similarity of a signal as a function of the time delay applied to it. 

### Fractal Time Series & The Hurst Exponent (H)
A **fractal time series** is one that exhibits statistical self-similarity, meaning its statistical character is the same at different timescales. This is the signature of a complex process with long-range correlations or "memory," where events in the distant past can influence the future.

The **Hurst exponent (H)** is a direct measure of this memory in a time series. It is defined by three regimes:
*   **H = 0.5**: Indicates a purely random process (a random walk) with no memory, where each step is independent of the last.
*   **0.5 < H ≤ 1.0**: Indicates a **persistent** or trend-reinforcing series. An increase is more likely to be followed by another increase, signifying positive long-range correlations and a system with memory.
*   **0 ≤ H < 0.5**: Indicates an **anti-persistent** or mean-reverting series. An increase is more likely to be followed by a decrease, signifying negative correlations.
