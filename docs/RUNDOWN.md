# Interspeech 2026 Narrative Flow & Cognitive Load Reduction Plan

**Core Philosophy:**
The goal of this restructuring is to eliminate "cognitive jumps"—moments where a reviewer asks "Why are we talking about this now?" or "How does this connect to the previous section?". We must guide the reviewer down a funnel: from the macroscopic problem (barrier attenuation) to the physical intuition (LDV anchor), then to the mathematical mechanism (Mass-Law Phase Shift -> GCC-PHAT), resulting in the engineering solution (PI-DNN), and finally concluding with the empirical proof (Confidently Wrong curve).

## Section-by-Section Paragraph Transitions

### 1. Introduction
*   **Paragraph 1 (The Problem):** Free-field DoA works great, but NLOS (through-barrier) fails because the barrier causes extreme transmission loss, destroying the SNR at the microphones.
*   **Paragraph 2 (The Villain):** *Transition:* Low SNR is hard, but *coherent interference* (a jammer) is deadly. Here we introduce the "Confidently Wrong" phenomenon. The microphones don't just fail; they confidently lock onto the jammer because it propagates freely in the air.
*   **Paragraph 3 (The Hardware Hero):** *Transition:* Algorithmic tweaks can't fix a physically destroyed signal. We must introduce a new modality. Enter the LDV. It measures the *structure-borne* vibration. *Crucial visual hook:* Explain that the LDV gets the signal *before* it enters the noisy room. This creates the "asymmetric" advantage.
*   **Paragraph 4 (The Algorithmic Hero):** *Transition:* Having the LDV signal isn't enough; raw STFT fusion fails because of the wall's transfer function (TF). Our solution: extract physical priors (cross-modal GCC-PHAT) and feed them into a Physics-Informed DNN (PI-DNN).
*   **Paragraph 5 (Contributions):** Standard bullet points summarizing the claims.

### 2. System Model
*   **Paragraph 1 (The Math Setup):** Define the signals formally. Equation 1 (Microphones) and Equation 2 (LDV).
*   **Paragraph 2 (Acoustic Impedance Mismatch - newly added previously):** *Transition:* Why does Equation 2 lack the jammer term $v(t)$? Because of the extreme acoustic impedance mismatch. Airborne jammers bounce off the heavy drywall, leaving the LDV channel pure. This answers the immediate "Wait, wouldn't the jammer vibrate the wall too?" question in the reviewer's head.

### 3. Proposed Physics-Informed Fusion Network
*   **3.1 Cross-Modal GCC Features (The Physical Mechanism):**
    *   *Paragraph 1:* Define the cross-spectrum $G_{Vm}(f)$. Show that its phase is corrupted by $H_{wall}(f)$.
    *   *Paragraph 2 (The "Aha!" Moment):* *Transition:* How do we fix this corrupted phase? **This is where we weave in the mathematical cancellation property.** We explain that the phase corruption $\angle H_{wall}(f)$ is structurally identical for both the LDV-MicL and LDV-MicR pairs. Therefore, when the network processes both spatial features, the shared wall transfer function mathematically cancels out in the differential delay domain ($\Delta\tau = \tau_{VR} - \tau_{VL}$). This mathematical identity guarantees generalization across diverse materials without the network needing to memorize specific acoustic impedances. *Cognitive Load Check:* We don't delve into complex dispersion physics; we point to a clean mathematical subtraction that the reviewer can immediately verify.
*   **3.2 DNN Architecture and Loss Formulation:**
    *   *Paragraph 1:* Describe concatenating the GCC-PHAT features and feeding them to the MLP.
    *   *Paragraph 2 (Preempting the PINN attack - newly added previously):* *Transition:* Wait, why is this called "Physics-Informed" if there's no PDE loss function? We proactively defend this: "Unlike classical PINNs that suffer from convergence instability via complex auxiliary losses, our architecture achieves physical compliance directly through its input domain." We justify using simple MSE loss.

### 4. Experimental Setup
*   **Paragraph 1:** Describe the drywall and equipment.
*   **Paragraph 2:** Describe the dataset, the SJR levels, and the PI-DNN training setup (16K parameters, real-time capable).

### 5. Experimental Results and Discussion
*   **5.1 High-Resolution Spatial Spectrum (The Visual Proof):**
    *   *Paragraph 1:* Before looking at error numbers, let's *look* at what the fusion does. Reference **Figure 3 (The MVDR Heatmap)**.
    *   *Paragraph 2:* Contrast the flat Mic-only line with the sharp LDV-Mic peak. This proves the LDV restores the spatial sparsity/resolution that was destroyed by the wall.
*   **5.2 Pure Barrier Accuracy vs Ablation Baseline:**
    *   *Paragraph 1:* Now quantify the error and algorithmic confidence. Reference **Table 1**, pointing out that traditional Mic-Mic SRP has a disastrous PSR (1.2dB), whereas the LDV Anchor restores it to a highly confident 8.5dB.
    *   *Paragraph 2 (Ablation - newly fixed previously):* *Transition:* Is the PI-DNN really necessary? What if we just used a CNN on raw STFTs? We explain the CNN raw STFT baseline (which fails at $6.5^\circ$ error because the dual-objective mapping of deconvolving the wall transfer function while estimating angle is ill-posed) compared to our 1.4° error. This proves the "Physics-Informed" GCC extraction is vital.
*   **5.3 Defeating the "Confidently Wrong" Phenomenon:**
    *   *Paragraph 1:* The ultimate test: the Jammer scenario. Reference **Figure 4 (The P_CW curve)**.
    *   *Paragraph 2:* Walk the reviewer through the graph. Point out how the Mic-only curve spikes to 95% error at negative SJRs, while the LDV-Mic PI-DNN curve stays pinned to the floor. *Transition:* Reiterate that the CNN (Raw STFT) baseline also struggles here (the middle curve in Fig 4), cementing our exact design choice.

### 6. Conclusions
*   Wrap up the story. Reiterate the narrative arc: Broken Mics -> LDV Anchor -> Mass-Law Physical Priors -> PI-DNN -> Defeating Jammers.

---
## Anticipated Reviewer Attacks & Our Pre-emptive Defenses

1.  **Attack:** "Why is this called Physics-Informed? There's no PDE loss!"
    *   **Defense:** Addressed in **Section 3.2**. We explicitly state we embed physics via the input domain to avoid the instability of PDE losses.
2.  **Attack:** "Doesn't the jammer vibrate the wall too? Your LDV equation ignores the jammer."
    *   **Defense:** Addressed in **Section 2.2**. We explicitly describe the "Acoustic Impedance Mismatch", explaining that airborne waves reflect rather than vibrating the heavy mass.
3.  **Attack:** "Your AI probably just memorized the sound of this specific drywall in your lab."
    *   **Defense:** Addressed in **Section 3.1**. We weave in the "Mass-Law / Coincidence Frequency / fixed 90-degree phase shift" physics, proving mathematically that the GCC-PHAT extraction is material-agnostic below the coincidence frequency.
4.  **Attack:** "Why use your feature extraction? Deep CNNs can learn anything from raw STFTs if given enough data."
    *   **Defense:** Addressed in **Section 5.2/5.3 and Table 1 / Fig 4**. We explicitly added the CNN (Raw STFT) ablation baseline, showing it fails to generalize and struggles against jammers despite having 100x more parameters.

## Conclusion
By following this exact narrative flow, the paper is no longer a collection of disconnected paragraphs. It is a tightly woven argument where every sentence anticipates and answers the reviewer's next logical question.