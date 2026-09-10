# MaleCNS Fly Brain: Multi-Channel Reservoir Computing (Liquid State Machine)

Turning a real biological brain into a chaotic time-series forecaster.

This project treats the connectome of the fruit fly (**MaleCNS v1.0** — 166,700 neurons, 88.3M sub-cellular nodes, ~151M synaptic connections) as the fixed recurrent weight matrix of a **reservoir computer** (a.k.a. an Echo State Network / Liquid State Machine)[cite: 1]. Instead of a randomly generated reservoir, the "brain" doing the computation is an actual, anatomically reconstructed nervous system, propagated on GPU as a sparse tensor[cite: 1]. A small linear readout layer is trained on top of it to forecast a chaotic time series (Mackey-Glass), from 1 step ahead out to 50 steps ahead[cite: 1].

The core question: **how much predictive structure is already sitting inside real neural wiring, before any learning happens in the recurrent connections at all?**

---

## How it works

1. **Connectome graph → recurrent reservoir.** The MaleCNS synaptic connectivity graph (source neuron, target neuron, synapse weight) is loaded and converted into a sparse `torch.sparse_coo_tensor` recurrent weight matrix `W_res`, scaled to a target spectral radius[cite: 1]. No connectome weights are trained — the biology stays fixed.

2. **Multi-channel sensory injection.** Rather than feeding the input signal into the network in one flat way, three anatomically distinct sub-populations of nodes are used as separate input channels[cite: 1]:
   - **Channel 1 — Optic Lobe (visual):** raw signal `u(t)`[cite: 1]
   - **Channel 2 — Antennal Lobe (olfactory):** temporal velocity `Δu(t) = u(t) − u(t−1)`[cite: 1]
   - **Channel 3 — Central Complex (mechanosensory):** non-linear acceleration term `u(t)² · sign(u(t))`[cite: 1]

   This gives the reservoir three complementary "views" of the driving signal, injected into three different neuropil regions, echoing how a real brain routes distinct sensory modalities into distinct anatomical regions.

3. **Reservoir dynamics.** State update is a standard leaky-integrator echo state network rule running on GPU[cite: 1]:
