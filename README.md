# IL-1β Signaling Model (PySB)

A compact, rule-based ODE model of IL-1β signaling and IL-1Ra antagonism built with **PySB (Python 3.9)**.  
The model reduces the pathway to **two reactions** and fits **IL-1Ra variant** kinetics (including **Isunakinra**) to **six dose–response curves** using **Nelder–Mead**.

---

## Overview

**Reactions**
1. `IL + R1  <->  IL_R1` (signaling complex; model output)
2. `A[v] + R1 <->  A_R1[v]` (non-signaling complex)

This yields **5 ODEs**, **4 parameters**, and **3 initial conditions** in the core ODEs system.  
**Only** variant-specific **kon/koff for IL-1Ra** are fitted; the rest remain fixed.

---

## Install

```bash
conda create -n IL_models python=3.10.9 -y
conda activate IL_models
pip install -r requirements.txt