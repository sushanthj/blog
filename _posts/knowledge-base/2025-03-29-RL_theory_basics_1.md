---
title: Crafting Neural Nets for RL
subtitle: RL Intro and Imitation Learning
featured_image: /images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg1_neural_net_policy.svg
categories: knowledge-base-reinforcement-learning-theory
permalink: /knowledge-base/reinforcement-learning-theory/basics_1/
---

* TOC
{:toc}

# Neural Nets ***are*** the 'Policy' 

![Neural net as policy](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg1_neural_net_policy.svg)

In the case of imitation learning or behaviour cloning this can be thought of as: **given expert demonstrations, can we train a neural network (a policy) to mimic those actions?** This turns out to be surprisingly nuanced. The way we design the model's outputs and loss function has a huge impact on how "expressive" the learned policy can be.

This post walks through the progression from naive approaches to more expressive generative policies.

---

# Discrete Actions: The Simple Case

Consider training a network to play Pacman. 

![Pacman policy](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg2_pacman_policy.svg)

This is just a **classification problem**. The model outputs 4 logits (one per action), and we train with **cross-entropy loss**. Simple, clean, and *maximally expressive* i.e. the softmax output can represent any distribution over those 4 buttons.

But what happens when actions are continuous...?

---

# Continuous Actions: Where Things Get Interesting

Now suppose we want to train a policy to estimate **steering angle** for a self-driving car. 

![Steering policy](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg3_steering_policy.svg)

## Attempt 1: Predict a Single Value - A Deterministic Policy

The simplest approach: make the model output a single angle and train with L2 (MSE) loss.

**The problem: mean-averaging.** Imagine we collect data from four expert drivers, all in the same state:

| Robot State | Expert Driver | Action (angle $\theta$) |
|:-----------:|:-------------:|:----------------------:|
| s | Driver_1 | -10&deg; |
| s | Driver_2 | +10&deg; |
| s | Driver_3 | -20&deg; |
| s | Driver_4 | +20&deg; |

With L2 loss, the model will learn to predict **0&deg;** --- the mean of all expert actions. 

![Mean averaging problem](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg4_mean_averaging.svg)

That's neither left nor right, which could be catastrophic if the experts were, say, swerving to avoid an obstacle.

Beyond mean-averaging, deterministic policies have a compounding error problem
1. The model is trained on expert data, so it learns to predict well on states that the expert visits.
2. But at test time, if the model makes a small mistake and ends up in a state that the expert never visited, it has no idea what to do

---

## Attempt 2: Predicting a Simple Distribution (Gaussian) - A Stochastic Policy

What if instead of predicting a single value, the model outputs the **parameters of a Gaussian distribution** --- a mean $\mu$ and variance $\sigma^2$?

![Gaussian policy](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg4b_gaussian_policy.svg)

- The expert data distribution for our steering example was **bimodal** (some drivers go left, some go right)
- A single Gaussian can only capture one mode
- If we train with L2 loss (which is equivalent to fitting the mean of a Gaussian), we again predict the average --- right between the two modes.

Even if we somehow get the Gaussian to match one of the modes, we're constraining the model to represent only a **unimodal** distribution. Not expressive enough.

## Attempt 3: Gaussian Mixture Model - Slightly More Expressive Stochastic Policy

What if we ask the model to predict a **mixture of Gaussians**? 

![GMM policy](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg5_gmm_policy.svg)

This is **more** expressive, but still limited by the number of mixture components we choose. If we pick 2 components, we can capture the bimodal distribution. But what if the expert data has 3 modes? Or 10 modes? We would need to arbitrarily choose a large number of components, which is inefficient and still may not capture the true distribution well.

---

## Attempt 4: Autoregressive Models: Even More Expressive Stochastic Policies

![Transformer with timestep outputs](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg6a_transformer_timesteps.svg)

Here we rely on discretizing the action space into small enough 'bins' such that we are as 'continous' as possible. (our best approximation)

![Bins to histogram](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg6b_bins_to_histogram.svg)

Each bin is again a probability (multi-class classification). More bins = more expressiveness, just like **larger vocabulary size in LLMs**.

## Training Stochastic Policies with Maximum Likelihood

What does it actually mean to "match the expert"? The model outputs a probability distribution over actions. When the expert acts, we check: **did our distribution put high probability where the expert acted?** If yes, the loss is low. If not, the loss is high, and training shifts the distribution toward expert actions.

<video width="100%" autoplay loop muted playsinline>
  <source src="/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/video_il_log_prob.mp4" type="video/mp4">
</video>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin: 2em 0;">
  <iframe src="https://drive.google.com/file/d/1MrgMteqzWWI-8XARpLUze6HmgaQhyHsO/preview" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
</div>

### Extension 1: Predicting Multiple Actions per State

![Autoregressive chain](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg8_autoregressive_chain.svg)

## Extension 2: Teacher Forcing (used exactly similarly in autoregressive language models)

**Teacher forcing** means that during training, we feed the *ground truth* $a_1$ (from the expert data) as input when computing $a_2$'s loss, rather than the model's own predicted $a_1$. This stabilizes training.

![Teacher forcing](/images/knowledge_base/concepts/reinforcement_learning_theory/imitation_learning/svg9_teacher_forcing.svg)

---

# Summary: The Expressiveness Ladder

| Approach | Output | Loss | Expressiveness |
|:---------|:-------|:-----|:---------------|
| Deterministic | Single value | L2 (MSE) | Lowest --- predicts the mean |
| Single Gaussian | $\mu, \sigma$ | Gaussian NLL | Unimodal only |
| Mixture of Gaussians | $\\{\mu_i, \sigma_i, w_i\\}$ | Mixture NLL | Multi-modal, fixed components |
| Autoregressive (discretized) | Bin probabilities | Cross-entropy | **Maximally expressive** |

The formal objective for expressive imitation learning:

$$\min_\theta \; -\mathbb{E}_{(\mathbf{s},\mathbf{a}) \sim \mathcal{D}}\left[\log \pi_\theta(\mathbf{a} \mid \mathbf{s})\right]$$

with an expressive distribution $\pi(\cdot \mid \mathbf{s})$. The more expressive the policy class, the better it can capture the full distribution of expert behavior.

---

# Appendix: The Reparameterization Trick

When training a Gaussian policy, we need gradients to flow through the sampling step. But sampling is stochastic --- you can't backpropagate through randomness.

The **reparameterization trick** solves this by rewriting:

$$z \sim \mathcal{N}(\mu, \sigma^2)$$

as a **deterministic function** of the parameters plus external noise:

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

Now $\mu$ and $\sigma$ are deterministic operations in the computation graph, and gradients flow through them cleanly:

$$\frac{\partial z}{\partial \mu} = 1, \qquad \frac{\partial z}{\partial \sigma} = \epsilon$$

In PyTorch, this is the difference between `dist.sample()` (no gradients) and `dist.rsample()` (reparameterized, gradients flow).

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin: 2em 0;">
  <iframe src="https://drive.google.com/file/d/1yqOP2Cw_PC78jlVJO7s8y0p4X0lcUCHx/preview" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
</div>

*Full animation: The Reparameterization Trick*

---

*Based on notes from Stanford CS224R (Spring 2025) --- Deep Reinforcement Learning.*
