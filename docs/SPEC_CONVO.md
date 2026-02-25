# USER

I want you to completely break down this [new paper](https://arxiv.org/pdf/2602.04770) on a new form of generative model architecture called "Drifting Models". I want a clear breakdown of what it is, how it works, how it compares to recent developments in the generative model space (diffusion model improvements, flow models, energy-based models, etc etc).

I want a breakdown of every point of novelty in the paper, with grounding in what came before it. 

Then, I want a complete and in-depth breakdown of what it would take to completely reproduce this paper, or at least create the source code for it. The source code and models for this paper have *not* been released yet, and there is no guarantee that they will be. This has been true of several past big models as well, like with Google's titans, alphafold 3, and several others. A github user named lucidrains has put in painstaking effort to completely replicate these mentioned architectures in torch, but he posted that one of them was his last release, so he may not be doing it this time. This time, I believe we will have to step into the shoes of lucidrains, and completely and perfectly replicate this architecture in pytorch.

I want a primary instructional focus on what it would take to replicate this. I know from some barebones skimming of the paper that it build on or makes use of google's TITANs architecture, and lucidrains *did* reproduce that one, along with a few others. We can utilize that work. 

Then, as a secondary focus, I want a detailed and categorical breakdown of what it would take to reproduce the experimental results, and whatever else comprises the paper.


Be as thorough as you possibly can. I expect this to be a **very long** answer (40+ “pages” as a rough unit; not literal pages). I am attaching the full paper, a markdown scan of the paper (with math, tables, content, etc), plus a zip of the extended scan with figures and all. I'm also attaching the zip of a [recent minimal reproduction](https://github.com/Algomancer/Minimal-Drifting-Models) of this architecture from GitHub.



# ASSISTANT

Below is a **deep, implementation-oriented teardown** of *Generative Modeling via Drifting* (arXiv: 2602.04770) and what it would take to **reproduce the architecture and results in PyTorch**—i.e., to “be lucidrains” for this one.

I’m grounding everything in the attached paper (PDF + scan) and the attached minimal reproduction zip. 

---

# 0. What you’re actually looking at

The paper introduces a new generative modeling paradigm:

* **Diffusion / flow matching:** do *iterative pushforward at inference time* (many NFEs), by integrating an SDE/ODE or stepping a denoiser.
* **Drifting Models:** do the *iterative pushforward at training time* (many SGD steps), and then run **one forward pass** at inference (1 NFE).

The core move is surprisingly simple:

1. Maintain a one-shot generator (f_\theta(\epsilon, \text{cond})) (e.g., a DiT).
2. Define a **drifting field** (V_{p,q}(x)) that tells you how a sample (x\sim q) should move to better match the data distribution (p).
3. Train the generator with a **stop-gradient fixed-point regression** so that generator outputs move in the direction of the drift under SGD.

Even more concretely:
**the optimizer’s parameter updates are treated as the “time axis”** along which distributions evolve.

That’s the novelty: instead of “simulate a dynamics at inference,” they “let SGD be the dynamics,” but still keep a *mathematically explicit vector field* that tells SGD what sample movement would reduce mismatch.

---

# 1. Drifting Models in one sentence (the mental model)

A Drifting Model is a **one-step generator** trained so that its samples have **zero drift** under a carefully designed **attraction–repulsion field** computed from **positive (real)** and **negative (generated + uncond)** samples (usually in a feature space).

When drift (\approx 0), the model is at equilibrium, and empirically (q \approx p).

---

# 2. The paper’s key conceptual inversion

## 2.1 Pushforward distribution “trajectory” lives in *training iterations*

They start with the usual generative objective:

[
q = f_# p_\epsilon
]

But instead of viewing (f) as fixed and applying many updates to (x) at inference, they consider training iteration (i):

* model (f_i)
* pushforward (q_i = [f_i]*# p*\epsilon)

As (i) increases, **the distribution (q_i) evolves**.

For a *fixed* noise draw (\epsilon), the sample changes because the model changes:

[
x_{i+1} = f_{i+1}(\epsilon)
\quad,\quad
\Delta x_i := f_{i+1}(\epsilon) - f_i(\epsilon)
]

So samples “drift” during training.

## 2.2 They formalize sample drift with a vector field

They posit a field (V_{p,q_i}(x_i)):

[
x_{i+1} = x_i + V_{p,q_i}(x_i)
]

where:

* (p) is the target (data) distribution,
* (q_i) is the current generated distribution.

The intended equilibrium property:

* when (p=q), samples stop drifting:
  [
  V_{p,p}(x) = 0
  ]

And they ensure that by enforcing an **anti-symmetry** condition:

[
V_{p,q}(x) = -V_{q,p}(x)
]

Then if (p=q), clearly (V_{p,p}(x) = -V_{p,p}(x)\Rightarrow V_{p,p}(x)=0).

This is their Proposition 3.1 in spirit.

---

# 3. The training objective is “BYOL/consistency-style” fixed-point regression

They write an equilibrium fixed-point equation:

[
f_{\hat\theta}(\epsilon)
========================

f_{\hat\theta}(\epsilon) + V_{p,q_{\hat\theta}}(f_{\hat\theta}(\epsilon))
]

and then train by the iteration rule:

[
f_{\theta_{i+1}}(\epsilon) \leftarrow f_{\theta_i}(\epsilon) + V_{p,q_{\theta_i}}(f_{\theta_i}(\epsilon))
]

implemented via a stop-grad target:

[
\mathcal{L}
===========

\mathbb{E}*\epsilon
\left[
\left|
f*\theta(\epsilon)
------------------

\texttt{stopgrad}\left(f_\theta(\epsilon) + V_{p,q_\theta}(f_\theta(\epsilon))\right)
\right|^2
\right]
]

This is the most lucidrains-relevant point:

### The gradient does *not* backprop through (V)

That’s why they use stop-grad: (V) depends on the distribution (q_\theta) (i.e., on the set of generated samples), and “backprop through a distribution” is messy. So they **treat drift as a frozen target movement** each step.

Also note the neat identity:

* If (x_\text{target} = \text{sg}(x + V)), then (x - x_\text{target} = -V) (value-wise)
* so the loss value equals (\mathbb{E}|V|^2)

And in Appendix C.2 they derive that the per-sample gradient w.r.t. (x) becomes:

[
\frac{\partial \mathcal{L}(x)}{\partial x} = -2V(x)
]

So the generator is literally trained so that its outputs move along the drift direction.

### What came before this (grounding)

This fixed-point-with-stopgrad template is very close in *training mechanics* to:

* **BYOL / SimSiam:** regression to a stop-grad target built from the same network (no negative samples).
* **Consistency models / iCT:** enforce consistency across time/noise levels with stop-grad targets.

But Drifting Models differ in the key source of the target:

* BYOL: target is a second network view
* consistency: target is a time-transformed sample under an SDE/ODE prior
* **drifting:** target is **(x + V_{p,q}(x))** where (V) is built from data-vs-model sample interactions.

---

# 4. The drifting field (V_{p,q}): mean-shift attraction minus repulsion

This is the other big novelty: they *explicitly construct* a drift field that becomes zero at match.

## 4.1 Generic form: expectation over positives and negatives

They start with:

[
V_{p,q}(x)= \mathbb{E}*{y^+\sim p}\mathbb{E}*{y^-\sim q}\left[K(x,y^+,y^-)\right]
]

and then instantiate (K) via attraction and repulsion terms inspired by **mean shift**.

## 4.2 Attraction and repulsion fields

Define:

[
V^+*p(x)=\frac{1}{Z_p(x)}\mathbb{E}*{y^+\sim p}[k(x,y^+)(y^+-x)]
]

[
V^-*q(x)=\frac{1}{Z_q(x)}\mathbb{E}*{y^-\sim q}[k(x,y^-)(y^--x)]
]

with normalizers:

[
Z_p(x)=\mathbb{E}*{y^+\sim p}[k(x,y^+)]
,\quad
Z_q(x)=\mathbb{E}*{y^-\sim q}[k(x,y^-)]
]

Then:

[
V_{p,q}(x)= V^+_p(x) - V^-_q(x)
]

Intuition:

* (V^+_p): “pull x toward nearby real samples”
* (V^-_q): “push x away from nearby generated samples”
* the subtraction ensures anti-symmetry, hence equilibrium.

## 4.3 Compact product form

They show that substituting yields:

[
V_{p,q}(x)
==========

\frac{1}{Z_p(x)Z_q(x)}
\mathbb{E}_{y^+\sim p,y^-\sim q}
\left[
k(x,y^+)k(x,y^-)(y^+ - y^-)
\right]
]

That form is what their compute routine implements (approximately).

---

# 5. The kernel choice and the “double softmax” trick

## 5.1 Kernel

They use:

[
k(x,y) = \exp\left(-\frac{1}{\tau}|x-y|\right)
]

Important detail: it’s (|x-y|) (not squared).

## 5.2 Normalization via softmax (“InfoNCE-like”)

In practice, they normalize kernel weights with a softmax over the sample axis. If you’ve implemented contrastive learning, this will look familiar:

* logits are (-|x-y|/\tau)
* softmax yields normalized weights

## 5.3 Extra normalization over the x-axis (the unusual part)

They also do an additional normalization over the set of (x)’s (over the batch dimension), which slightly improves performance.

Implementation-wise they:

1. compute row-softmax (over y)
2. compute col-softmax (over x)
3. take geometric mean:

[
A = \sqrt{A_\text{row}\odot A_\text{col}}
]

This resembles “make it more doubly-stochastic-ish” and is reminiscent of Sinkhorn-ish scaling, but cheaper.

---

# 6. Algorithm 2: the actual compute_V you must replicate

The paper’s Algorithm 2 is the canonical implementation (the minimal repo you attached is basically a direct toy version of this).

Given:

* (x): generated batch ([N,D])
* (y_\text{pos}): real positives ([N_\text{pos},D])
* (y_\text{neg}): negatives ([N_\text{neg},D]) (often includes x itself)

Steps:

1. pairwise distances:

   * (dist_\text{pos} = cdist(x,y_\text{pos}))
   * (dist_\text{neg} = cdist(x,y_\text{neg}))
2. if (y_\text{neg} = x), mask diagonal with huge number to ignore self
3. logits: (-dist/T)
4. concatenate logits ([N, N_\text{pos}+N_\text{neg}])
5. compute:

   * row-softmax over y axis
   * col-softmax over x axis
   * (A = \sqrt{A_\text{row}A_\text{col}})
6. split (A) back into (A_\text{pos}, A_\text{neg})
7. compute weights:

   * (W_\text{pos} = A_\text{pos}\cdot \sum A_\text{neg})
   * (W_\text{neg} = A_\text{neg}\cdot \sum A_\text{pos})
8. drift:

   * drift(*\text{pos}= W*\text{pos}y_\text{pos})
   * drift(*\text{neg}= W*\text{neg}y_\text{neg})
9. return (V = drift_\text{pos} - drift_\text{neg})

This is **not** MMD gradient, not GAN discriminator gradient, not score matching. It’s a **kernelized attraction–repulsion field**.

---

# 7. Why anti-symmetry matters (and how they empirically show it)

Their Table 1 is a huge warning sign for reproduction:

If you break anti-symmetry (e.g., scale attraction or repulsion differently), it collapses.

Examples:

* attraction-only: FID explodes
* scale either term: FID explodes

So when you re-implement, don’t “improve” this by tuning attraction/repulsion separately unless you want the ablation behavior.

**Implementation takeaway:**
Your drift computation must preserve the (V_{p,q}=-V_{q,p}) structure. Small tweaks can break equilibrium.

---

# 8. Feature-space drifting: required for high-dimensional realism

This is one of the most practically important parts and also one of the hardest to reproduce.

## 8.1 The core issue

Their method depends on the kernel (k(x,y)). In high-dimensional raw spaces, naive L2 distances are garbage:

* everything is far
* kernels go flat
* drift signal vanishes

They explicitly say they **could not make ImageNet work without a feature encoder**, even with VAE latents.

So they compute drift not on (x), but on (\phi(x)), where (\phi) is a pretrained encoder.

## 8.2 Feature-space drifting loss

Single feature:

[
\mathbb{E}\left[\left|\phi(x)-\text{sg}\left(\phi(x)+V(\phi(x))\right)\right|^2\right]
]

Multiple features:

[
\sum_j \mathbb{E}\left[\left|\phi_j(x)-\text{sg}\left(\phi_j(x)+V(\phi_j(x))\right)\right|^2\right]
]

where (\phi_j) are multi-scale, multi-location features.

### Important implementation detail about gradients

You need gradients from the loss to flow back to the generator through (\phi) (and VAE decoder if you decode). But since the target is stop-grad, you **do not need gradients through the drift field computation** itself.

A very practical PyTorch design is:

* compute (\phi(x)) **with grad**
* compute (\phi(y)) for positives/negatives **without grad**
* compute drift (V) from **detached** features
* build stop-grad target

This gives the correct (\partial L/\partial x) direction while saving a ton of memory.

---

# 9. Multi-scale, multi-location feature set (A.5): the “secret sauce” complexity

This is one of the biggest reproduction burdens.

They do not just drift on a single pooled embedding. For each ResNet feature map (H_i\times W_i\times C_i), they construct multiple types of feature vectors:

For each feature map:

**(a)** per-location vectors: (H_iW_i) vectors in (\mathbb{R}^{C_i})
**(b)** global mean + global std: 2 vectors in (\mathbb{R}^{C_i})
**(c)** (2\times2) patch mean + std: (2\cdot (H_i/2)(W_i/2)) vectors
**(d)** (4\times4) patch mean + std: (2\cdot (H_i/4)(W_i/4)) vectors

Additionally, from the **encoder input** (H_0W_0C_0), they compute:

* per-channel mean of squared values (x^2), giving one (C_0)-dim vector.

Then they say:

* compute one drifting loss per such vector
* sum all losses
* plus also include “vanilla drifting loss without (\phi)” (more on that ambiguity later)

They show that richer feature sets improve FID (A.5 ablation table).

### Reproduction cost warning

This is potentially the *dominant* cost in implementation complexity and memory engineering.

Even if each drift is small, the number of feature vectors is huge (especially per-location vectors at higher resolutions), and you’re doing kernel computations per feature.

If you want to reproduce results, you need:

* careful batching
* chunking across locations
* no-grad for real features
* probably activation checkpointing for (\phi(x))

---

# 10. Feature & drift normalization (A.6): another must-copy detail

They normalize **each feature** so distances are on consistent scales.

## 10.1 Feature normalization

For each feature (\phi_j \in \mathbb{R}^{C_j}):

[
\tilde{\phi}_j = \phi_j / S_j
]

They choose (S_j) so average distance becomes (\sqrt{C_j}):

[
\mathbb{E}*{x}\mathbb{E}*{y}
\left[
|\tilde{\phi}_j(x)-\tilde{\phi}_j(y)|
\right]
\approx \sqrt{C_j}
]

by setting:

[
S_j = \frac{1}{\sqrt{C_j}}\mathbb{E}*{x}\mathbb{E}*{y}\left[|\phi_j(x)-\phi_j(y)|\right]
]

They compute this from batch pairwise distances, and **stop-grad (S_j)**.

Then temperature is scaled as:

[
\tilde{\tau}_j = \tau\sqrt{C_j}
]

This is subtle but important: it makes (\tau) more feature-invariant.

## 10.2 Drift normalization

They also normalize drift magnitudes per feature:

[
\tilde{V}_j = V_j / \lambda_j
]

with:

[
\lambda_j = \sqrt{\mathbb{E}\left[\frac{1}{C_j}|V_j|^2\right]}
]

Then the per-feature loss is:

[
L_j=\text{MSE}\Big(
\tilde{\phi}_j(x),
\text{sg}(\tilde{\phi}_j(x)+\tilde{V}_j)
\Big)
]

Again: this is crucial if you sum thousands of terms.

## 10.3 Multiple temperatures

They compute drift fields for (\tau\in{0.02,0.05,0.2}), normalize each, sum them:

[
\tilde{V}*{j} = \sum*{\tau} \tilde{V}_{j,\tau}
]

This reduces sensitivity to the “right” scale.

---

# 11. Their CFG is *training-time* and is not standard CFG (but related)

This part matters a lot because it affects both training logic and inference interface.

## 11.1 Standard diffusion CFG (what came before)

In diffusion:

* you run the model twice (cond and uncond)
* combine predictions at inference with guidance scale
* costs ~2× NFEs per step (unless you do tricks)

## 11.2 Drifting CFG formulation

They define a *negative distribution* for class (c):

[
\tilde{q}(\cdot|c) = (1-\gamma)q_\theta(\cdot|c) + \gamma p_\text{data}(\cdot|\varnothing)
]

They train so that (\tilde{q}(\cdot|c)=p_\text{data}(\cdot|c)).

Solving gives:

[
q_\theta(\cdot|c)=\alpha p(\cdot|c) - (\alpha-1)p(\cdot|\varnothing)
]
with (\alpha = \frac{1}{1-\gamma}\ge 1).

That’s analogous to CFG’s “extrapolation” idea, but baked into training by mixing negatives.

## 11.3 How they implement it (A.7): unconditional real negatives with weight (w)

During training:

* you include (N_\text{unc}) unconditional real samples (random classes) as **extra negatives**
* you weight them by a factor (w) in kernel computation

They derive the effective negative mixture:

[
\tilde{q}(\cdot|c)
==================

\frac{(N_\text{neg}-1)q_\theta(\cdot|c) + N_\text{unc}w,p_\text{data}(\cdot|\varnothing)}
{(N_\text{neg}-1)+N_\text{unc}w}
]

So:

[
\alpha = \frac{(N_\text{neg}-1)+N_\text{unc}w}{N_\text{neg}-1}
]

Given desired (\alpha), compute:

[
w = \frac{(\alpha-1)(N_\text{neg}-1)}{N_\text{unc}}
]

They also apply this same weighting in the **global distance normalization**.

## 11.4 Inference

At inference time:

* you provide (\alpha) to the network as a conditioning input
* you still run only one forward pass (1-NFE)

So they get “CFG-like behavior” without extra NFE.

---

# 12. Architecture: it’s DiT, but with a bundle of modern Transformer tweaks

The generator is a DiT-like transformer (f_\theta(\epsilon, c, \alpha)), with:

* latent generation: input/output (32\times 32\times 4), patch size 2 → 16×16=256 tokens
* pixel generation: output (256\times 256\times 3), patch size 16 → 16×16=256 tokens

They list:

* **SwiGLU**
* **RoPE**
* **RMSNorm** (they write RM-SNorm; it’s essentially RMSNorm)
* **QK-Norm**

And conditioning via:

* AdaLN-zero (DiT standard)
* plus “in-context conditioning tokens”

## 12.1 In-context conditioning tokens (register tokens)

They prepend **16 tokens** “for in-context conditioning.”
These tokens are formed by:

* project the conditioning vector (from (c,\alpha) plus style embeddings)
* add positional embeddings
* prepend to the patch sequence

This resembles “register tokens / prefix tokens” used in a few modern DiT variants.

## 12.2 Random style embeddings (important to reproduce)

They add 32 discrete “style tokens”:

* sample 32 random integers in ([0,64))
* embed them (learned table of size 64)
* sum the 32 embeddings
* add to the conditioning vector

This is a **secondary noise source** (discrete) beyond Gaussian.

They report improved FID with this.

---

# 13. Feature encoder: a custom ResNet-style MAE trained in latent space

This is another major novelty / reproduction burden.

## 13.1 Why: they want multi-scale conv features in latent space

If you used pixel-space SimCLR/MoCo, for latent models you’d need:

* decode latents to pixels
* run pixel encoder
* backprop through decoder (expensive)

Instead they train a ResNet MAE directly on SD-VAE latents: **latent-MAE**.

## 13.2 Architecture

* Encoder: ResNet-style conv net, GN instead of BN
* Decoder: U-Net-ish deconv decoder
* Masking: zero out masked patches (not remove tokens)
* Input shapes:

  * latent MAE: (32\times32\times4)
  * pixel MAE: patchify (256\times256\times3) into (32\times32\times192) with (8\times8) patches, then same ResNet

They describe it like ResNet-34-ish (basic blocks, [3,4,6,3])—but their Table 8 has an inconsistency (it says “bottleneck” for one config). You should treat block type as an **ambiguity to resolve experimentally**.

## 13.3 MAE masking scheme

* work on a 32×32 grid (latent or patchified pixels)
* mask 2×2 patches with 50% probability
* implement by **zeroing** those patches

## 13.4 Training

* L2 reconstruction loss on masked regions
* AdamW lr (4\times10^{-3})
* batch size 8192 (!)
* EMA decay 0.9995
* random resized crop augmentation
* for latent: augment images before VAE-encoding to latents

## 13.5 Classification fine-tuning

They then fine-tune the MAE with a linear classifier head for 3k steps using:

[
\lambda L_\text{cls} + (1-\lambda)L_\text{recon}
]

with (\lambda) warmup 0→0.1 over 1k steps.

This “MAE + cls ft” gives a big improvement.

---

# 14. System-level results (what they claim)

Key headline:

* ImageNet 256×256, latent protocol: **FID 1.54** at 1 NFE (DiT-L/2 generator)
* pixel protocol: **FID 1.61** at 1 NFE (DiT-L/16 generator)

They compare to:

* multi-step diffusion/flows
* single-step diffusion/flow methods trained from scratch (MeanFlow, iMeanFlow, etc.)
* GANs in pixel space

They claim state-of-the-art among 1-NFE methods, and competitive vs multi-step.

Also interesting:

* their best latent model’s optimal FID is at (\alpha=1) (which is “no CFG” in diffusion terms), but CFG still changes IS/FID tradeoff similarly.

---

# 15. Relationship to other generative model families (with real grounding)

This is where a lot of people will misclassify Drifting Models. Here’s the clean taxonomy.

## 15.1 Diffusion models and recent “step reduction” work

**Diffusion:** define an SDE/ODE in “noise level time” (t), train a score/velocity field, integrate at inference.
Step reduction methods fall into:

1. **distillation:** teacher multi-step → student few-step/one-step
2. **train-from-scratch few/one-step:** bake the dynamics into training losses

**Drifting Models are neither.**
They do not model an SDE/ODE in (t), and they do not approximate an inference trajectory. There is no (t) variable in the model.

Instead:

* the dynamics is “SGD iteration (i)”
* the vector field is computed from sample interactions at each step

So comparisons:

* diffusion step reduction tries to preserve a known trajectory family
* drifting defines a new, training-driven trajectory family

## 15.2 Flow matching / rectified flows

Flow matching learns a continuous-time velocity field to transport noise to data.

Similarities:

* both use a **vector field**
* both can be seen as distribution transport

Differences:

* flow matching: vector field is parameterized and integrated at inference
* drifting: vector field is **computed nonparametrically from sample sets** and only used as a training target; inference is direct mapping.

A good way to say it:

> Flow matching learns the field; drifting *uses a field to learn the mapping*.

## 15.3 GANs

GANs:

* one-step generator
* discriminator defines gradient signal via min-max

Drifting:

* one-step generator
* no discriminator
* “critic” is replaced by explicit kernel interactions between positives and negatives

The drift field is like a handcrafted, nonparametric “critic gradient.”

Practical consequence:

* drifting avoids adversarial instability
* but depends heavily on feature space and kernel quality

## 15.4 Normalizing flows

NFs are invertible, likelihood-based.

Drifting:

* not invertible
* no likelihood
* “prior → data” only, not “data → prior”

So it’s much closer to GAN/DiT diffusion-family architectures than to NFs.

## 15.5 Energy-based models

EBMs:

* define energy (E(x))
* train so data has low energy, model samples high energy
* sampling requires MCMC / Langevin, typically many steps

Drifting:

* no explicit energy
* no MCMC at inference
* training uses positive/negative samples (like EBMs), but updates generator directly

However, conceptually:

* attraction to data + repulsion from model resembles EBM learning signals
* compute_V feels like a particle-interaction gradient field

So it’s spiritually closer to particle EBMs / contrastive divergence than to diffusion.

## 15.6 MMD / moment matching

They explicitly discuss MMD in Appendix C.2.

If you view MMD loss as a discrepancy (L(p,q)), then (-\partial L/\partial x) is a drift field.

But:

* MMD yields an unnormalized kernel gradient drift
* their method depends on **normalized kernels**, drift normalization, flexible step sizes, CFG extension, etc.
* they say they couldn’t get reasonable results with MMD directly

So drifting is a “V-centric generalization” of that family.

## 15.7 SVGD / particle-based VI and mean-shift

This is arguably the closest “mathematical vibe.”

* Mean shift: move points to modes of a KDE via weighted neighbor mean.
* SVGD: move particles with kernel attraction to target + repulsion among particles.

Their (V_p^+ - V_q^-) has exactly that flavor:

* data pulls samples
* model samples repel
* kernel defines neighborhood

The big difference:

* SVGD updates particles directly
* drifting updates *a generator network* so that its outputs behave like particles under this field

This is a key conceptual bridge for grounding novelty: they’re essentially embedding a particle interaction dynamic into generator training via stop-grad regression.

---

# 16. Every point of novelty (with “what came before”)

Here’s the novelty list I’d use if I were writing a reproduction README.

## Novelty A — Paradigm: training-time distribution evolution as the generative “dynamics”

**What came before:** inference-time dynamics (diffusion/flows), or adversarial training (GANs), or distillation from a teacher trajectory.

**New:** treat the SGD training path (f_i) as a sequence of pushforward distributions (q_i), and define an explicit sample update rule that SGD is induced to follow.

## Novelty B — Drift field design with explicit equilibrium via anti-symmetry

**What came before:** discriminator gradients (GAN), score gradients (diffusion), MMD gradients, etc.

**New:** a drift field (V_{p,q}) that is explicitly anti-symmetric and thus guaranteed to be 0 when (p=q). The training objective is literally minimizing drift magnitude.

## Novelty C — “Stopgrad drift regression” objective

**What came before:** BYOL/SimSiam stop-grad regression, consistency models.

**New:** use stop-grad regression where the target is *the sample drift computed from p-vs-q interactions*, not from augmentations or SDE time steps.

## Novelty D — Practical kernelized mean-shift with the “double softmax + geometric mean” normalization

**What came before:** plain kernel normalization, InfoNCE row-softmax.

**New:** extra softmax over x-axis and geometric mean (A=\sqrt{A_\text{row}A_\text{col}}), plus the weight construction that preserves equilibrium.

## Novelty E — Feature-space drifting with massive multi-scale multi-location feature set + per-feature normalization

**What came before:** perceptual losses (paired), CLIP-guided things, pooled feature matching.

**New:** unpaired feature-space drift, computed at many local features, with careful normalization of both features and drift so losses can be summed without collapsing.

## Novelty F — Training-time CFG via *unconditional real negatives* and network-conditioned (\alpha)

**What came before:** inference-time CFG requiring 2 passes per step (cond/uncond) in diffusion.

**New:** CFG emerges from *negative sampling design* at training time; inference remains 1 pass and (\alpha) is just an input.

## Novelty G — latent-MAE (ResNet MAE trained directly on SD-VAE latents) as the feature encoder

**What came before:** using pixel SSL encoders, or VAE latents but with pixel encoders after decoding.

**New:** pretrain a conv MAE in latent space to cheaply get semantic-ish multi-scale features without decoding.

## Novelty H — random discrete “style embedding” noise for one-step DiT generators

**What came before:** StyleGAN style mixing; discrete latents in some token models.

**New:** treat the input noise distribution as joint continuous+discrete; add style embeddings to conditioning.

---

# 17. Reproducing the paper’s *architecture* in PyTorch (primary focus)

Now the big part: what it takes to implement this “for real.”

I’ll split this into:

1. **Minimal faithful core** (the algorithm)
2. **Full ImageNet-grade implementation** (the painful details)
3. **Engineering to make it train**

---

## 17.1 Minimal faithful core (what the “Minimal-Drifting-Models” repo already has)

Your attached minimal repo does a toy 2D version and implements:

* compute_V with softmax normalizations
* stopgrad regression
* drift = attraction - repulsion

That’s the correct core.

But to match the paper, you need:

* multi-feature drift loss
* feature encoder(s)
* CFG negative mixing (weighted negatives)
* DiT generator details (register tokens, style embeddings, QK-Norm, RMSNorm, RoPE)
* large-scale batching and queues

---

## 17.2 The “must-implement” modules for a faithful reproduction

If I were building this like lucidrains (clean modules, minimal glue), I’d implement the following.

### Module 1 — Drift field `compute_V`

Requirements:

* implements Algorithm 2
* supports:

  * multiple temperatures
  * optional extra x-axis normalization
  * diagonal self-mask for generated negatives
  * unconditional negative weighting (w) (CFG)
  * batched groups (per-class groups)

Design details:

* Shapes:

  * group dimension (G=N_c)
  * (x): ([G, N_\text{neg}, D])
  * (y_\text{pos}): ([G, N_\text{pos}, D])
  * (y_\text{neg}): ([G, N_\text{neg} + N_\text{unc}, D])

* Efficiency:

  * use `torch.cdist` for distance
  * use bf16/fp16 logits
  * chunk if D is huge

CFG weighting:

* create weights over y columns: `w_pos=1`, `w_neg_generated=1`, `w_unc=w`
* weighting in kernel softmax is best implemented as **adding (\log w)** to logits for uncond columns (equivalent to multiplying kernel values by w before normalization)

### Module 2 — Feature normalization and drift normalization

You must compute:

* (S_j) from mean pairwise distances (with stop-grad)
* (\lambda_j) from mean squared drift magnitude

This is a key stability component.

### Module 3 — Multi-scale multi-location feature extraction

This is the “big one.”

You need:

* an encoder that can return intermediate feature maps
* a function that converts each feature map into a list of (C_i)-dim vectors:

  * per-location
  * global mean/std
  * patch mean/std (2×2 and 4×4)
  * plus channel second moments for input

And you must:

* compute one drifting loss per vector
* sum them

Critical:

* they normalize across spatial locations by concatenating all locations in a map to estimate scales; i.e., don’t compute (S_j) separately per location.

### Module 4 — Generator: DiT with their modifications

Must include:

* patchify/unpatchify
* AdaLN-Zero conditioning
* register tokens (16 in-context tokens)
* style embeddings (32 random indices from codebook size 64) added to conditioning
* RMSNorm, RoPE, QK-Norm, SwiGLU

Unknowns you must decide:

* exact alpha embedding mechanism
* exact RoPE variant
* exact QK-Norm implementation details

But the general version is reproducible.

### Module 5 — CFG sampling and negative mixing

You need to replicate:

* training-time alpha sampling distribution (Table 8)
* conversion alpha → w
* negative set construction: generated negatives + unconditional real negatives
* ensure your diagonal mask means “(N_\text{neg}-1)” effective generated negatives

### Module 6 — Feature encoder: latent-MAE and pixel-MAE

To match paper results you likely need:

* their ResNet-style MAE encoder
* pretraining schedule (huge)
* classification fine-tune schedule

Or else you will not reach their FIDs.

### Module 7 — Sample queues (A.8)

They used MoCo-style queues:

* per-class queue size 128
* global uncond queue size 1000
* push 64 new real samples per step
* sample without replacement

You need to implement this or use a data loader that can sample:

* (N_\text{pos}) real samples from class c
* (N_\text{unc}) unconditional random samples

Queue is easier in distributed training, because it reduces dataloader complexity.

---

## 17.3 Practical clarifications and “paper ambiguities” you must resolve

There are a few places where the paper is ambiguous or inconsistent, and a faithful reproduction must pick an interpretation.

### Ambiguity 1 — Table 8 generator input size for pixel models

Table 8 lists generator input size as (32\times32\times4) even for pixel-space models, but the main text says pixel-space generator outputs (256\times256\times3). I would treat Table 8 as a typo for pixel generator input size.

**Reproduction plan:** implement pixel generator as 256×256×3 with patch size 16.

### Ambiguity 2 — ResNet MAE block type inconsistency

* Table 3: latent-MAE uses **basic** blocks.
* Table 8: one column claims **bottleneck**.

Appendix A.3 text describes basic blocks.

**Reproduction plan:**

* implement both basic-block and bottleneck-block variants behind a config flag
* start with basic blocks (because A.3 and Table 3 match)
* optionally test bottleneck if you’re chasing final decimal points

### Ambiguity 3 — “vanilla drifting loss without φ” for pixel-space

Computing drift in raw pixel space is extremely expensive if interpreted literally as (D=256\cdot256\cdot3).

**Reproduction plan:**

* interpret “vanilla without φ” as drift on the generator’s *native output space* (latent or pixel) but do it on a **downsampled** representation or **patchified input** (32×32 grid) to keep cost reasonable
* or implement it but make it optional / ablate

You can confirm by ablation: if disabling it barely changes FID, it wasn’t essential.

---

# 18. A faithful PyTorch implementation plan (lucidrains-style)

This section is a blueprint for how I would actually build the repo.

## 18.1 Repository structure

A clean implementation might look like:

```
drifting_models/
  __init__.py
  drift_field.py          # compute_V, kernels, CFG weighting
  drift_loss.py           # multi-temp, normalization, batching
  features.py             # multi-scale/multi-location vectorization
  generator_dit.py        # DiT + register tokens + style embeddings
  rope.py                 # rotary embeddings
  norms.py                # RMSNorm, QKNorm
  mae_resnet.py           # ResNet-style MAE encoder (+ optional decoder)
  queues.py               # per-class + global queues
  data/
    imagenet_latents.py   # precomputed SD-VAE latents dataset
    imagenet_pixels.py
  train/
    train_latent_imagenet.py
    train_pixel_imagenet.py
  eval/
    sample.py
    fid_is.py
    nn_clip.py
configs/
  drifting_latent_b2.yaml
  drifting_latent_l2.yaml
  drifting_pixel_b16.yaml
  drifting_pixel_l16.yaml
```

## 18.2 The generator: exact pieces to implement

### Inputs

* continuous noise (\epsilon)
* class label (c)
* CFG strength (\alpha)
* discrete style indices (s\in[0,64)^{32})

### Conditioning vector

* (emb_c): learned class embedding
* (emb_\alpha): embed scalar alpha (use Fourier features + MLP, or log-alpha + MLP)
* (emb_s): sum of 32 style embeddings (64-entry codebook)
* `cond = emb_c + emb_alpha + emb_s`

### In-context/register tokens

* `cond_token = Linear(cond)` → ([B, d])
* create 16 tokens:

  * token_i = cond_token + pos_embed_i
* prepend these to patch tokens

### Patchify noise

Latent model:

* input tensor ([B, 4, 32, 32])
* patch size 2 → patches ([B, 256, 16])?? Actually patch dim is (2*2*4=16)
* project to hidden dim (768/1024)

Pixel model:

* input ([B, 3, 256, 256])
* patch size 16 → patch dim (16*16*3=768)
* project to hidden dim

### Transformer blocks

Each block needs:

* RMSNorm or LayerNorm? They say RM-SNorm (RMSNorm)
* attention with:

  * QK-Norm: normalize query and key vectors per head
  * RoPE applied to Q/K
* MLP with SwiGLU

### AdaLN-Zero

Standard DiT uses:

* modulate normalization parameters per block using `cond`
* gating residuals with zero init so the network starts near identity

You must replicate AdaLN-Zero precisely enough.

---

## 18.3 Drift loss: the exact computation graph you want

This is the most important “do it the paper way” part:

1. Generate x = f(eps, c, alpha, style)
2. Extract (\phi(x)) **with gradients**
3. Extract (\phi(y^+)), (\phi(y^\text{unc})) **without gradients**
4. Construct y_neg = [x (detached), y_unc]
5. For each feature j:

   * compute S_j (detach)
   * compute drift V_j from detached features
   * compute lambda_j
   * build target (detach)
   * compute MSE between normalized (\phi_j(x)) and target
6. Sum over all features and temps
7. Backprop to generator parameters only

This gives:

* generator gradients
* no backprop through V computation
* no backprop through y features
* but backprop through (\phi) for x only (input-gradient only)

---

## 18.4 How to implement unconditional weighting (w) correctly

If you have logits over y columns:

* concatenated logits: `[pos_logits | neg_logits_generated | neg_logits_uncond]`

Then:

* for uncond columns, add `log(w)` **before** softmax.

Also apply `w` in computing mean distance normalization (S_j) (weighted mean over y columns).

---

## 18.5 Batch construction (their “grouped” computation)

They sample (N_c) class labels and run Alg. 1 per class, so your batch should be grouped:

* labels: ([N_c])
* expand each label to N_neg generated samples

So:

* (B = N_c\cdot N_\text{neg})

But drift computations must be per-class-group, the interactions do not mix across classes.

In tensor form:

* x: ([N_c, N_\text{neg}, ...])
* y_pos: ([N_c, N_\text{pos}, ...])
* y_unc: ([N_c, N_\text{unc}, ...]) or ([N_c,N_\text{unc}]) drawn from a global queue

This matters: if you accidentally mix classes in drift field, you’re changing the method.

---

# 19. What it would take to reproduce their *experimental results* (secondary focus)

This is mostly a compute + pipeline question.

---

## 19.1 ImageNet latent-space (FID 1.54) reproduction checklist

### Data pipeline

1. Obtain ImageNet-1k training + val.
2. Use “standard SD-VAE tokenizer”:

   * encode images to latents
   * likely store latents to disk
3. Store latents and labels:

   * training latents: ~1.28M
   * val latents: 50k

### Feature encoder pretraining: latent-MAE

To match their best feature encoder:

* base width 640
* masking ratio 50%
* pretrain 1280 epochs
* batch size 8192
* AdamW lr 4e-3
* EMA 0.9995
* then classification finetune 3k steps

This is a large-scale pretraining run; it’s arguably *the* gatekeeping step for reproducing results.

### Generator training

Use Table 8 settings:

* DiT-L/2
* 1280 epochs
* 200k steps
* B=8192 generated per step
* N_pos=64 for L/2 (per Table 8)
* N_neg=64
* N_c=128
* N_unc=32
* temps {0.02,0.05,0.2}
* AdamW lr 4e-4, betas (0.9,0.95)
* warmup 10k
* grad clip 2.0
* EMA decays tried: {0.999, 0.9995, 0.9998, 0.9999}
* CFG alpha sampling: 50% alpha=1, 50% p(alpha)∝alpha^-3, alpha in [1,4]
* inference alpha search [1.0,3.5]

### Evaluation protocol

* generate 50k images from val classes
* decode latents with VAE decoder
* compute FID vs ImageNet val set
* compute IS similarly

Also:

* run nearest-neighbor analysis with CLIP cosine similarity vs training set (paper does this qualitatively)

---

## 19.2 ImageNet pixel-space (FID 1.61) reproduction checklist

This is harder.

### Feature encoders

They report that using the same MAE recipe as latent gives bad FID; they need:

* ResNet + ConvNeXt-V2 as feature encoders
* both MAE-pretrained and fine-tuned

So you need:

* pixel-MAE pretraining (same recipe but on patchified pixels)
* ConvNeXt-V2 MAE weights (either train or use public weights, but must match their version)

### Generator

* DiT-L/16
* 640 epochs (they trained less due to time)
* patch size 16
* output 256×256×3
* drift computed in pixel feature space

### Evaluation

* generate 50k images
* compute FID / IS

---

## 19.3 Toy experiments reproduction

This is the easiest and already matched by your minimal repo.

You need:

* 2D target mixture distribution
* small MLP generator
* compute_V on raw coordinates
* show convergence under different initializations
* show anti-symmetry ablation fails

---

## 19.4 Robotics control reproduction

They follow Diffusion Policy protocol and replace the diffusion generator with drifting generator:

* compute drift loss on raw action sequences / representations
* no feature encoder needed

To reproduce you need:

* Diffusion Policy codebase / environments
* same tasks: Lift, Can, ToolHang, PushT, BlockPush, Kitchen
* implement Drifting Policy model and training objective
* match evaluation protocol: success averaged over last 10 checkpoints

---

# 20. Engineering reality: what will break first if you try to reproduce

If you try to reproduce paper-quality results, these are the top likely failure points.

## 20.1 Feature encoder mismatch

If your feature encoder doesn’t produce:

* meaningful neighborhoods
* multi-scale features
* stable normalization stats

your kernel will go flat and training will stall or collapse.

This is consistent with their claim that ImageNet fails without feature encoder.

## 20.2 Drift computation scaling

Multi-location features produce huge numbers of drift computations.
You’ll need:

* chunking over locations
* batched `cdist`
* mixed precision
* careful memory management

## 20.3 CFG weighting correctness

If you apply (w) incorrectly (or forget to apply it in normalization), the effective mixture distribution differs from their derivation, and guidance behavior will differ.

## 20.4 Anti-symmetry violations from “small” refactors

It’s easy to break the equilibrium condition by:

* normalizing pos/neg differently
* applying different temps incorrectly
* mixing groups incorrectly
* forgetting diagonal self-mask
* using asymmetric kernels

Table 1 shows this is catastrophic.

---

# 21. A concrete “step into lucidrains’ shoes” reproduction roadmap

If you want the best chance of success, here’s a staged approach.

## Stage 0 — Validate core drift field

* reproduce toy 2D with your minimal repo
* verify:

  * loss equals ||V||^2
  * drift goes to ~0 when distributions match
  * anti-symmetry ablation breaks it

## Stage 1 — CIFAR-10 / ImageNet-64 latent

* use a small VAE or pretrained autoencoder
* implement DiT-B with register tokens + style embeddings
* implement drift in feature space using a fixed encoder (e.g., a public SimCLR)
* verify training converges and produces reasonable samples

## Stage 2 — Implement full multi-scale/multi-location features + normalization

* port feature extraction exactly as A.5/A.6
* verify ablation table trend: richer features improves FID

## Stage 3 — Train latent-MAE encoder

* implement ResNet MAE and pretrain on your latents
* confirm Table 3 trend: stronger encoder → better FID

## Stage 4 — Scale to ImageNet 256 latent and match Table 5

* match Table 8 hyperparams
* match alpha sampling
* evaluate FID

## Stage 5 — Pixel-space

* add ConvNeXt-V2 feature encoder and combine with ResNet features
* train pixel generator and chase Table 6

---

# 22. A few “implementation design choices” that matter more than you’d think

These are things that are not emphasized in the paper, but will matter in your PyTorch reproduction.

## 22.1 How you embed alpha

The paper says the network is conditioned on (\alpha), but not exactly how.

You should:

* embed (\alpha) similarly to how DiT embeds timesteps:

  * Fourier features of (\alpha) or (\log\alpha)
  * MLP → hidden dim
* add it to class embedding

## 22.2 Freezing the feature encoder

They say gradients backprop through encoder, but they don’t explicitly say they update it during generator training.

I strongly suspect:

* encoder weights are frozen
* gradients flow through encoder to x (input-grad only)

This is how perceptual-feature training is typically done.

**In PyTorch:**
Set encoder params `requires_grad=False`, but don’t wrap the forward in `no_grad()` for generated x.

## 22.3 Where “vanilla loss without φ” lives

If you interpret it as raw pixel drift, you’ll suffer.

Better:

* compute identity-feature drift in the encoder input domain (e.g., patchified 32×32×C0) at most
* or omit in pixel-space and see if FID changes

---

# 23. How to verify your implementation is *correct* (unit tests / invariants)

These are the tests I’d write before burning compute.

## 23.1 compute_V invariants

1. **Equilibrium:** if y_pos == y_neg and distributions match, drift sums to ~0 (up to sampling noise).
2. **Anti-symmetry:** compute V_{p,q}(x) and V_{q,p}(x) and confirm they negate (with same batch content).
3. **Diagonal mask:** ensure self-negative has near-zero weight.
4. **Scaling:** if you scale all features by s, after feature normalization you should get similar drift.

## 23.2 Training objective correctness

* verify loss value equals mean(||V||^2) for a batch (numerically)
* verify gradient w.r.t x equals -2V approximately (finite difference check)

## 23.3 CFG behavior sanity checks

* increasing inference (\alpha) should increase IS-like behavior and eventually worsen FID-like behavior (qualitative)
* verify alpha=1 corresponds to w=0 (or effectively no unconditional negatives)

---

# 24. How your attached “Minimal-Drifting-Models” repo fits in

Your zip is a good “kernel-level” sanity check.

It correctly mirrors:

* stopgrad update
* compute_V with normalization
* toy distribution matching

What it doesn’t cover (and what you’ll need to add):

* grouping by class
* CFG negative mixing and alpha conditioning
* multi-feature extraction and normalization
* DiT generator and training stability
* feature encoder pretraining and usage

Think of it as: **Algorithm 2 validated**, but **everything around it is the hard part**.

---

# 25. Bottom line: what is “hard” about reproducing this paper?

Not the drift idea—it’s clean.

The hard parts are:

1. **Feature encoder quality** (latent-MAE pretraining + fine-tuning)
2. **Multi-scale/multi-location drift loss implementation** with normalization
3. **Massive batching** and careful memory management
4. **Exact CFG negative weighting** and alpha-conditioning details
5. **Transformer implementation parity** (DiT + modern tweaks)

If you implement only the core drift in raw space, you’ll reproduce toys but not ImageNet.

---

If you want, I can also (in the same style) map this to a **precise PyTorch pseudocode training loop** with tensor shapes at every line (grouped batch, drift computation per-feature, normalization, alpha→w weighting) and a checklist of config values to mirror Table 8—so you can drop it into a lucidrains-like single-file implementation.

But even without that follow-up, everything above should be enough to begin a faithful “no official code” reproduction effort.



