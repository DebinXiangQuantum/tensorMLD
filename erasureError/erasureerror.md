Based on the mathematical frameworks provided in **TensorMLD** and the **Spin Glass Reflection**  papers, the extension to erasure errors (common in neutral atom platforms) relies on modifying the statistical weight of specific qubits.

While the provided papers focus on depolarizing noise (where the error location is unknown), the "Essential Math" for extending this to erasure errors (where the error location is *known*) involves treating the erasure as a failure with maximal uncertainty (), which collapses the complex tensor terms into simple identities.

### 1. The Mathematical Extension

The core extension requires adjusting the **error weight parameter**  derived from the physical error rate .

#### Step 1: Probability Assignment for Erasures

In neutral atom platforms, an erasure error corresponds to a qubit that is lost or detected in a leakage state. Unlike a depolarizing error (unknown location), an erasure is **heralded** (known location).

* **Standard Error:**  (small probability).
* **Erasure Error:** The decoder knows an error occurred at index  but not its Pauli type (X or Z). Thus, the probability of the specific Pauli error at that location is maximized:



#### Step 2: Weight Transformation ()

The TensorMLD paper defines the weight  (the stiffness of the bond in the spin-glass model) using the log-likelihood ratio of the error probability:


Substituting the erasure probability () into this equation eliminates the weight:


#### Step 3: Tensor Simplification ()

In the tensor network, the "Energy Tensor"  encodes the probability of error. Its general form is given as:


When  (erasure):

* 
* 

Thus, the tensor for an erased qubit collapses to a scalar identity:


### 2. Results and Physical Interpretation

Applying this math yields the following results for the spin-glass and tensor network models:

* 
**Spin Glass "Broken Bonds":** In the spin-glass Hamiltonian , setting  effectively removes the term. This corresponds to "breaking the bond" in the lattice. Physically, this means there is **zero energy penalty** for assigning an error to an erased qubit. The decoder is free to flip this spin to satisfy the syndromes without incurring a "cost."


* **Tensor Network Pruning:** Because the tensors for erased qubits become , they act as pass-throughs. They do not increase the complexity of the contraction in the same way stiff bonds do. This theoretically simplifies the contraction for those specific indices, as the network effectively has "holes" where the erasures are.
* 
**Neutral Atom Applicability:** For neutral atoms, where connectivity is long-range and erasures are dominant, this extension allows the TensorMLD decoder to prioritize correcting "unknown" errors (depolarizing) while automatically filling in the "known" blanks (erasures) with the value that best satisfies the global stabilizer constraints.