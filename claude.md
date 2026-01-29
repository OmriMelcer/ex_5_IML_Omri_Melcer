# Claude Code Guidelines - IML Exercise 5

## Project Overview

University assignment for Introduction to Machine Learning (IML) - Exercise 5.
Implements and analyzes probabilistic models (GMM, UMM) and a small-scale GPT2 transformer for language modeling, using the Europe countries dataset and Shakespeare text corpus.

**Student:** Omri Melcer (ID: 208880211)

## Role

Claude serves as a **study helper** - not a code writer. The student writes all code. Claude assists with:

- **Bug debugging** - help identify and explain bugs when asked
- **Code review** - review code for correctness, style, and adherence to exercise requirements
- **Graph/plot creation** - write plotting code when asked
- **Report writing** - help draft the final PDF report (max 8 pages)
- **Concept explanations** - explain GMM, UMM, self-attention, transformers, and related math

**Do NOT write implementation code unless the student explicitly asks.**

## Plotting Rules

When writing graph/plotting code:

- **Always save figures to disk** (`plt.savefig()`), never call `plt.show()`
- **Combine as many plots as possible into a single image** using subplots (`fig, axes = plt.subplots(...)`)
- **Every plot must have:** a clear title, labeled axes, and a legend where applicable
- Use `plt.tight_layout()` or `fig.suptitle()` + `plt.subplots_adjust()` to avoid overlapping labels
- Save to a `plots/` directory with descriptive filenames (e.g., `gmm_samples_k10.png`)
- Use `dpi=150` or higher for readable resolution

## Project Structure

```
ex_5_IML/
├── README.md                 # Student info (name, username, ID)
├── Exercise_5_2026.pdf       # Full exercise specification
├── pyproject.toml            # Project config (Python 3.13+)
├── main.py                   # Entry point (placeholder)
├── dataset.py                # Data handling (EuropeDataset, ShakespeareDataset, DataHandler) - PROVIDED, do not modify
├── mixture_models.py         # GMM and UMM skeleton code - STUDENT IMPLEMENTS
├── transformer.py            # GPT2 transformer skeleton code - STUDENT IMPLEMENTS
├── train.csv                 # Europe geographic data (~107k rows: index, lon, lat, country)
├── test.csv                  # Europe geographic test data (~13k rows)
├── train_shakespeare.txt     # Shakespeare training text (~38k lines)
└── test_shakespeare.txt      # Shakespeare test text (~2k lines)
```

## Exercise Structure

### Part 1: Mixture Models (mixture_models.py)

**Gaussian Mixture Model (GMM):**
- Implement: `forward()`, `loss_function()`, `sample()`, `conditional_sample()`
- Optimized parameters: `self.logits`, `self.means`, `self.log_variances`
- Use `torch.logsumexp` for numerical stability
- Use `nn.functional.log_softmax` for log p(k)
- 2D diagonal covariance only
- Train with n_components = [1, 5, 10, n_classes]

**Uniform Mixture Model (UMM):**
- Same structure as GMM but with uniform distributions
- Optimized parameters: `self.logits`, `self.centers`, `self.log_sizes`
- Use -1e6 instead of -inf for out-of-support log probabilities
- Can use `torch.distributions.Uniform` for sampling

### Part 2: Transformer (transformer.py)

- Implement `CausalSelfAttention` (multi-head with causal mask)
- Train GPT model on Shakespeare data
- Generate sentences (default: start with "the ", generate 30 chars)
- Experiment with top-k sampling (k=5)

## Key Technical Constraints

- **Allowed libraries only:** numpy, matplotlib, torch, tqdm
- **Seeding:** `np.random.seed(42)` and `torch.manual_seed(42)` (reproducibility matters, exact seed value flexible)
- **Data normalization:** Normalize Europe data to zero mean, unit variance per dimension
- **Default parameters:** Always use the defaults provided in the skeleton code unless told otherwise

## Report Requirements

- PDF format, max 8 pages
- Named `ex5_208880211.pdf`
- Must include all requested plots
- Every answer must include an explanation (no marks for unexplained answers)
- Answer questions in sequential order

## Submission

- Zip file: `ex5_208880211.zip`
- Contains: README.md, .py files, report PDF
- No extra files (no data files, no plots directory)